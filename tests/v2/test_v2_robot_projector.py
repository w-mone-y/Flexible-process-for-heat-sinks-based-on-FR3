from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brazing_sim.dual_line import DualLineRuntime, DualLineSceneAdapter, UnifiedV2Runtime, UnitStage
from brazing_sim.flexible import build_custom_plan, build_inline_plan
from brazing_sim.motion import HOME_QPOS

ROOT = Path(__file__).resolve().parents[2]
V2_XML = ROOT / "scenes" / "production" / "brazing_line_v2.xml"


def _direction_reversal_ratio(samples: list[tuple[int, np.ndarray]]) -> float:
    reversals = 0
    eligible = 0
    for index in range(2, len(samples)):
        waypoint, current = samples[index]
        previous_waypoint, previous = samples[index - 1]
        older_waypoint, older = samples[index - 2]
        if waypoint != previous_waypoint or waypoint != older_waypoint:
            continue
        first_delta = previous - older
        second_delta = current - previous
        eligible += 1
        reversals += int(np.any(first_delta * second_delta < -1.0e-8))
    return 0.0 if eligible == 0 else reversals / eligible


def _quaternion_excursion_rad(quaternions: list[np.ndarray]) -> float:
    reference = quaternions[0]
    dots = np.abs(np.stack(quaternions) @ reference)
    return float(np.max(2.0 * np.arccos(np.clip(dots, -1.0, 1.0))))


def _finger_inner_gap_m(adapter: DualLineSceneAdapter, arm_name: str) -> float:
    open_gap = {"arm1": 0.042, "arm3": 0.042}[arm_name]
    positions = [
        float(
            adapter.data.qpos[
                adapter.model.jnt_qposadr[adapter.model.joint(f"v2_{arm_name}_{side}_finger_joint").id]
            ]
        )
        for side in ("left", "right")
    ]
    return open_gap - sum(positions)


def _geom_world_aabb(
    adapter: DualLineSceneAdapter,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative visible-world AABB for primitive and mesh geoms."""

    size = np.asarray(adapter.model.geom_size[geom_id], dtype=float).copy()
    geom_type = int(adapter.model.geom_type[geom_id])
    if geom_type == 2:  # sphere
        size = np.repeat(size[0], 3)
    elif geom_type == 3:  # capsule
        size = np.asarray([size[0], size[0], size[1] + size[0]], dtype=float)
    elif geom_type == 5:  # cylinder
        size = np.asarray([size[0], size[0], size[1]], dtype=float)
    rotation = np.asarray(adapter.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    half_extent = np.abs(rotation) @ size
    center = np.asarray(adapter.data.geom_xpos[geom_id], dtype=float).copy()
    return center, half_extent


def _signed_aabb_clearance_m(
    adapter: DualLineSceneAdapter,
    left_geom_id: int,
    right_geom_id: int,
) -> float:
    left_center, left_half = _geom_world_aabb(adapter, left_geom_id)
    right_center, right_half = _geom_world_aabb(adapter, right_geom_id)
    gaps = np.abs(left_center - right_center) - left_half - right_half
    if np.any(gaps > 0.0):
        return float(np.linalg.norm(np.maximum(gaps, 0.0)))
    return float(np.max(gaps))


def _signed_xy_aabb_clearance_m(
    adapter: DualLineSceneAdapter,
    left_geom_id: int,
    right_geom_id: int,
) -> float:
    """Return signed top-view clearance, independent of a skim in Z."""

    left_center, left_half = _geom_world_aabb(adapter, left_geom_id)
    right_center, right_half = _geom_world_aabb(adapter, right_geom_id)
    gaps = np.abs(left_center[:2] - right_center[:2]) - left_half[:2] - right_half[:2]
    if np.any(gaps > 0.0):
        return float(np.linalg.norm(np.maximum(gaps, 0.0)))
    return float(np.max(gaps))


@pytest.mark.parametrize(
    ("preset", "expected_count", "expected_spacing_m"),
    (("A", 5, 0.0475), ("B", 4, 0.075), ("C", 7, 0.031)),
)
def test_v2_fin_supply_inventory_matches_v1_order_count_and_consumption(
    preset: str,
    expected_count: int,
    expected_spacing_m: float,
) -> None:
    """A branch shows exactly the current kit and retires each picked blank."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        order_id = f"RAW_KIT_{preset}"
        runtime.submit_order(preset, order_id=order_id)
        observed_initial = False
        observed_consumed = False
        for _ in range(8_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            unit = runtime.units[f"{order_id}_UNIT_01"]
            if unit.branch is None:
                continue
            arm_name = "arm1" if unit.branch.value == "ARM1_A" else "arm3"
            branch = "a" if arm_name == "arm1" else "b"
            state = adapter.robot_motion_snapshot()[arm_name]
            if ":INSTALL_FIN:" not in str(state["operation"]):
                continue
            fixed_visible = sum(
                float(adapter.model.geom(f"v2_fin_{branch}_raw_fin_{index:02d}").rgba[3]) > 0.5
                for index in range(1, 13)
            )
            proxy_visible = float(adapter.model.geom(f"v2_{arm_name}_raw_fin_proxy_geom").rgba[3]) > 0.5
            visible_inventory = fixed_visible + int(proxy_visible)
            physical_handoff = int(bool(state["installed_fin_revealed"]))
            assert visible_inventory == expected_count - int(unit.fins_installed) - physical_handoff
            authored_y = [
                float(adapter.model.geom(f"v2_fin_{branch}_raw_fin_{index:02d}").pos[1])
                for index in range(1, expected_count + 1)
            ]
            np.testing.assert_allclose(
                np.diff(authored_y),
                expected_spacing_m,
                atol=1.0e-9,
            )
            assert 0.5 * (authored_y[0] + authored_y[-1]) == pytest.approx(0.0)
            observed_initial |= unit.fins_installed == 0
            observed_consumed |= unit.fins_installed >= 1
            if observed_initial and observed_consumed:
                break
        assert observed_initial
        assert observed_consumed
    finally:
        adapter.close()


def test_v2_arm3_picks_successive_fins_from_successive_magazine_slots() -> None:
    """The next B-line fin must remain in its authored slot until pickup."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    sampled_sources: dict[int, np.ndarray] = {}
    try:
        runtime.submit_order("A", order_id="ARM1_LEAD_A")
        runtime.submit_order("B", order_id="ARM3_SLOTS_B")
        for _ in range(12_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            unit = runtime.units["ARM3_SLOTS_B_UNIT_01"]
            state = adapter.robot_motion_snapshot()["arm3"]
            if unit.branch is None or unit.branch.value != "ARM3_B":
                continue
            if ":INSTALL_FIN:" not in str(state["operation"]):
                continue
            index = state["fin_index"]
            if not isinstance(index, int) or index not in {1, 2}:
                continue
            if state["workpiece_held"] or int(state["waypoint_index"]) > 1:
                continue
            proxy = np.asarray(adapter.data.body("v2_arm3_raw_fin_proxy").xpos, dtype=float).copy()
            authored = np.asarray(
                adapter.data.geom(f"v2_fin_b_raw_fin_{index:02d}").xpos,
                dtype=float,
            ).copy()
            # The free proxy can settle by a few tenths of a millimetre under
            # its active weld; this is still two orders of magnitude smaller
            # than the 75 mm B-magazine slot pitch.
            np.testing.assert_allclose(proxy[:2], authored[:2], atol=2.0e-3)
            sampled_sources.setdefault(index, proxy)
            if set(sampled_sources) == {1, 2}:
                break

        assert set(sampled_sources) == {1, 2}
        assert np.linalg.norm(sampled_sources[2] - sampled_sources[1]) >= 0.02
    finally:
        adapter.close()


def test_v2_arm3_completes_every_fin_lifecycle_on_the_parallel_branch() -> None:
    """A lead Arm1 order must leave the complete second fin kit to Arm3."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    lifecycle = {index: set() for index in range(1, 5)}
    arm3_failure_seen = False
    try:
        runtime.submit_order("A", order_id="ARM1_LEAD_FOR_FULL_ARM3")
        runtime.submit_order("B", order_id="ARM3_FULL_LIFECYCLE")
        unit = runtime.units["ARM3_FULL_LIFECYCLE_UNIT_01"]
        for _ in range(12_000):
            runtime.tick(0.02)
            adapter.sync(runtime)
            adapter.step_physics(0.02)
            state = adapter.robot_motion_snapshot()["arm3"]
            if ":INSTALL_FIN:" in str(state["operation"]):
                arm3_failure_seen |= bool(state["failure"])
                index = state["fin_index"]
                assert isinstance(index, int) and index in lifecycle
                if state["grasp_verified"] and state["workpiece_held"]:
                    lifecycle[index].add("picked")
                label = str(state["target_zh"])
                if "纯Z下降" in label or "纯Z向下放置" in label:
                    lifecycle[index].add("inserted")
                if state["release_verified"]:
                    lifecycle[index].add("opened")
                    assert unit.tray_id is not None
                    assert adapter.component_visible(unit.tray_id, f"fin_{index:02d}")
                    lifecycle[index].add("placed")
            if runtime.complete and adapter.transport_settled:
                break

        assert unit.branch is not None
        assert unit.branch.value == "ARM3_B"
        assert unit.fins_installed == unit.fin_count == 4
        assert not arm3_failure_seen
        expected = {"picked", "inserted", "opened", "placed"}
        assert all(stages == expected for stages in lifecycle.values()), lifecycle
        assert runtime.complete
        assert adapter.transport_settled
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("preset", "expected_thickness_m"),
    (("A", 0.0020), ("C", 0.0018)),
)
def test_v2_gripper_stops_at_the_current_fin_thickness(
    preset: str,
    expected_thickness_m: float,
) -> None:
    """The fingers must contact the two fin faces, never cross through them."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        order_id = f"CLAMP_{preset}"
        runtime.submit_order(preset, order_id=order_id)
        observed = False
        for _ in range(4_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            unit = runtime.units[f"{order_id}_UNIT_01"]
            arm_name = "arm1" if unit.branch is None or unit.branch.value == "ARM1_A" else "arm3"
            state = adapter.robot_motion_snapshot()[arm_name]
            if ":INSTALL_FIN:" not in str(state["operation"]) or not state["workpiece_held"]:
                continue
            assert _finger_inner_gap_m(adapter, arm_name) == pytest.approx(
                expected_thickness_m,
                abs=0.00015,
            )
            observed = True
            break
        assert observed, f"{preset}型没有进入夹紧持有阶段"
    finally:
        adapter.close()


def test_v2_both_fin_grippers_release_visibly_without_touching_adjacent_fins() -> None:
    """Both branches open clearly in-slot while retaining neighbour clearance."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("C", order_id="RELEASE_ARM1_C")
        runtime.submit_order("A", order_id="RELEASE_ARM3_A")
        observed: dict[str, float] = {}
        for _ in range(18_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            for arm_name in ("arm1", "arm3"):
                if arm_name in observed:
                    continue
                operation = runtime.operations.get(arm_name.upper())
                state = adapter.robot_motion_snapshot()[arm_name]
                if (
                    operation is None
                    or operation.kind != "INSTALL_FIN"
                    or state["fin_index"] != 2
                    or not state["release_verified"]
                ):
                    continue
                expected_gap_m = float(state["fin_thickness_m"]) + 0.0020
                assert float(state["finger_inner_gap_m"]) == pytest.approx(
                    expected_gap_m,
                    abs=0.00025,
                )
                unit = runtime.units[operation.unit_id]
                assert unit.tray_id is not None
                neighbour_id = int(adapter.model.geom(f"{unit.tray_id.lower()}_fin_01").id)
                clearances = [
                    _signed_aabb_clearance_m(
                        adapter,
                        int(adapter.model.geom(f"v2_{arm_name}_gripper_{side}").id),
                        neighbour_id,
                    )
                    for side in ("left", "right")
                ]
                minimum_clearance_m = min(clearances)
                assert minimum_clearance_m >= 0.0035
                observed[arm_name] = minimum_clearance_m
            if len(observed) == 2:
                break

        assert set(observed) == {"arm1", "arm3"}
    finally:
        adapter.close()


def test_v2_base_pick_changes_colour_and_uses_a_vertical_sampled_descent() -> None:
    """Mirror V1's suction feedback and slow, pose-locked final placement."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=False)
    adapter = DualLineSceneAdapter(V2_XML)
    before_colour: np.ndarray | None = None
    held_colours: list[np.ndarray] = []
    descent_positions: list[np.ndarray] = []
    try:
        runtime.submit_order("A", order_id="BASE_VISUAL_A")
        suction_pad = adapter.model.geom("v2_arm1_suction_pad")
        for _ in range(5_000):
            runtime.tick(0.02)
            adapter.sync(runtime)
            adapter.step_physics(0.02)
            state = adapter.robot_motion_snapshot()["arm1"]
            if ":BASE_LOADING:" not in str(state["operation"]):
                continue
            if not state["workpiece_held"]:
                before_colour = np.asarray(suction_pad.rgba, dtype=float).copy()
            if state["workpiece_held"]:
                held_colours.append(np.asarray(suction_pad.rgba, dtype=float).copy())
            if (
                "逐步纯Z下降" in str(state["target_zh"]) or "缓慢放置基板" in str(state["target_zh"])
            ) and state["workpiece_held"]:
                descent_positions.append(np.asarray(state["actual_tcp_position_m"], dtype=float).copy())
            if state["release_verified"]:
                break

        assert before_colour is not None
        assert held_colours
        assert max(float(np.linalg.norm(colour[:3] - before_colour[:3])) for colour in held_colours) >= 0.25
        assert len(descent_positions) >= 12
        positions = np.stack(descent_positions)
        assert float(np.ptp(positions[:, 0])) <= 0.003
        assert float(np.ptp(positions[:, 1])) <= 0.003
        assert float(positions[0, 2] - positions[-1, 2]) >= 0.025
        assert float(np.max(np.diff(positions[:, 2]))) <= 0.001
        assert float(np.max(np.abs(np.diff(positions[:, 2])))) <= 0.005
    finally:
        adapter.close()


def test_v2_carried_base_uses_the_short_direct_aisle_to_s1() -> None:
    """The full base stays clear while following lift/straight/drop motion."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="BASE_AISLE_A")
        proxy_id = int(adapter.model.geom("v2_arm1_raw_base_proxy_geom").id)
        beam_id = int(adapter.model.geom("v2_arm1_tool_rack_beam").id)
        minimum_clearance_m = float("inf")
        held_samples = 0
        transfer_positions: list[np.ndarray] = []
        transfer_labels: set[str] = set()
        for _ in range(4_000):
            runtime.tick(0.005)
            adapter.sync(runtime)
            adapter.step_physics(0.005)
            state = adapter.robot_motion_snapshot()["arm1"]
            if state["workpiece_held"]:
                label = str(state["target_zh"])
                if "携板" in label:
                    transfer_positions.append(
                        np.asarray(
                            state["actual_tcp_position_m"],
                            dtype=float,
                        ).copy()
                    )
                    transfer_labels.add(label)
                # ``mj_geomDistance`` may return a false zero for separated
                # rotated box/box pairs.  World-aligned bounds are more
                # conservative here: a positive gap proves the real OBBs are
                # separated, while still catching any possible overlap.
                proxy_half = (
                    np.abs(adapter.data.geom_xmat[proxy_id].reshape(3, 3))
                    @ adapter.model.geom_size[proxy_id, :3]
                )
                beam_half = (
                    np.abs(adapter.data.geom_xmat[beam_id].reshape(3, 3))
                    @ adapter.model.geom_size[beam_id, :3]
                )
                axis_gaps = (
                    np.abs(adapter.data.geom_xpos[proxy_id] - adapter.data.geom_xpos[beam_id])
                    - proxy_half
                    - beam_half
                )
                conservative_clearance = (
                    float(np.linalg.norm(np.maximum(axis_gaps, 0.0)))
                    if np.any(axis_gaps > 0.0)
                    else float(np.max(axis_gaps))
                )
                minimum_clearance_m = min(
                    minimum_clearance_m,
                    conservative_clearance,
                )
                held_samples += 1
            if state["release_verified"]:
                break

        assert held_samples >= 10
        assert minimum_clearance_m >= 0.020
        assert transfer_positions
        assert transfer_labels == {"S1 携板直线移动至托盘上方"}
        transfer = np.stack(transfer_positions)
        # There is no lateral escape or return: X stays on the shared pickup
        # and placement centreline while Y advances monotonically toward S1.
        assert float(np.ptp(transfer[:, 0])) <= 0.010
        assert float(np.max(np.diff(transfer[:, 1]))) <= 0.002
    finally:
        adapter.close()


def test_thin_base_descends_to_support_without_penetration_or_release_float() -> None:
    """The in-flight proxy and settled pool must share one thickness-aware pose."""

    pytest.importorskip("mujoco")
    plan = build_custom_plan(
        order_id="THIN_BASE_HANDOFF",
        quantity=1,
        priority=10,
        product={
            "base_size_m": [0.36, 0.22, 0.002],
            "fin_size_m": [0.30, 0.002, 0.06],
            "fin_count": 5,
            "fin_pitch_m": 0.02,
            "path_margin_m": 0.015,
            "path_width_m": 0.004,
            "nozzle_spacing_m": 0.005,
            "nozzle_tip_height_m": 0.004,
            "material_speed_m_s": 0.04,
            "target_clamping_force_n": 20.0,
            "recipe": "demo_brazing",
        },
    )
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    proxy_bottoms: list[float] = []
    suction_alignment_positions: list[np.ndarray] = []
    try:
        runtime.submit_plan(plan)
        proxy_id = int(adapter.model.geom("v2_arm1_raw_base_proxy_geom").id)
        unit = runtime.units["THIN_BASE_HANDOFF_UNIT_01"]
        for _ in range(6_000):
            runtime.tick(0.005)
            adapter.sync(runtime)
            adapter.step_physics(0.005)
            state = adapter.robot_motion_snapshot()["arm1"]
            if "缓慢吸取基板" in str(state["target_zh"]) and not state["workpiece_held"]:
                suction_alignment_positions.append(
                    np.asarray(adapter.data.geom_xpos[proxy_id], dtype=float).copy()
                )
            if state["workpiece_held"] and (
                "逐步纯Z下降" in str(state["target_zh"]) or "缓慢放置基板" in str(state["target_zh"])
            ):
                proxy_bottoms.append(
                    float(adapter.data.geom_xpos[proxy_id, 2] - adapter.model.geom_size[proxy_id, 2])
                )
            if state["release_verified"]:
                break

        assert unit.tray_id is not None
        settled_name = f"{unit.tray_id.lower()}_base_plate"
        settled = adapter.data.geom(settled_name)
        settled_bottom = float(settled.xpos[2] - adapter.model.geom(settled_name).size[2])
        assert len(proxy_bottoms) >= 8
        assert min(proxy_bottoms) >= settled_bottom - 0.0005
        assert proxy_bottoms[-1] == pytest.approx(settled_bottom, abs=0.0005)
        alignment = np.stack(suction_alignment_positions)
        alignment_steps = np.linalg.norm(np.diff(alignment, axis=0), axis=1)
        assert float(np.linalg.norm(alignment[-1] - alignment[0])) >= 0.0005
        assert float(np.max(alignment_steps)) <= 0.0002
    finally:
        adapter.close()


def test_v2_completed_branch_b_tray_clears_arm3_before_merge() -> None:
    """A completed S3B pallet must not pass through Arm3 on its way downstream."""

    mujoco = pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="CLEARANCE_LEAD_A")
        runtime.submit_order("B", order_id="CLEARANCE_BRANCH_B")
        arm3_geoms = []
        for geom_id in range(int(adapter.model.ngeom)):
            name = mujoco.mj_id2name(
                adapter.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            if (
                name
                and (name.startswith("arm3_") or name.startswith("v2_arm3_"))
                and int(adapter.model.geom_contype[geom_id]) > 0
            ):
                arm3_geoms.append(geom_id)

        minimum_clearance_m = float("inf")
        checked_samples = 0
        for _ in range(30_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            unit = runtime.units["CLEARANCE_BRANCH_B_UNIT_01"]
            transfer = adapter.transport_snapshot().get(unit.tray_id or "")
            if (
                unit.tray_id is not None
                and transfer is not None
                and transfer["moving"]
                and transfer["source"]
                in {
                    "INSTALL_B",
                    "MERGE_B_WAIT",
                    "MERGE",
                    "S4",
                    "BUFFER_1",
                    "BUFFER_2",
                    "BUFFER_3",
                    "FURNACE",
                    "POST_SCAN",
                }
            ):
                tray_geom = int(adapter.model.geom(f"{unit.tray_id.lower()}_geom").id)
                for arm_geom in arm3_geoms:
                    fromto = np.zeros(6, dtype=float)
                    minimum_clearance_m = min(
                        minimum_clearance_m,
                        float(
                            mujoco.mj_geomDistance(
                                adapter.model,
                                adapter.data,
                                tray_geom,
                                arm_geom,
                                2.0,
                                fromto,
                            )
                        ),
                    )
                checked_samples += 1
            if unit.stage.value == "COMPLETE":
                break

        assert checked_samples >= 10
        # ``mj_geomDistance`` reports exact negative penetration for the
        # original bug but may clamp separated convex pairs to zero.  The
        # static swept-centreline contract supplies the 320 mm layout margin;
        # this physical replay locks down the user-visible no-penetration
        # invariant.
        assert minimum_clearance_m >= -1.0e-6
    finally:
        adapter.close()


def test_v2_complete_outbound_payloads_clear_s2b_and_arm3() -> None:
    """Replay both S3 exits with the complete fixture, not only the tray slab."""

    mujoco = pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    minimum_table_clearance = {"ARM1_A": float("inf"), "ARM3_B": float("inf")}
    minimum_b_south_cell_clearance = float("inf")
    minimum_arm3_clearance = float("inf")
    maximum_transport_plane_error = 0.0
    checked_samples = {"ARM1_A": 0, "ARM3_B": 0}
    try:
        runtime.submit_order("A", order_id="FULL_SWEEP_A")
        runtime.submit_order("B", order_id="FULL_SWEEP_B")
        s2b_top = int(adapter.model.geom("v2_station_s2b_top").id)
        b_south_obstacles = (
            int(adapter.model.geom("v2_station_s2a_top").id),
            int(adapter.model.geom("v2_fin_table_b_top").id),
        )
        arm3_geoms = [
            geom_id
            for geom_id in range(int(adapter.model.ngeom))
            if (
                (
                    name := mujoco.mj_id2name(
                        adapter.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    )
                )
                and (name.startswith("arm3_") or name.startswith("v2_arm3_"))
                and int(adapter.model.geom_contype[geom_id]) > 0
            )
        ]

        for _ in range(30_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            transports = adapter.transport_snapshot()
            for unit in runtime.units.values():
                if unit.tray_id is None or unit.branch is None:
                    continue
                transport = transports.get(unit.tray_id)
                if (
                    transport is None
                    or float(transport["progress"]) < 0.02
                    or transport["source"]
                    not in {
                        "INSTALL_A",
                        "INSTALL_B",
                        "MERGE_A_WAIT",
                        "MERGE_B_WAIT",
                    }
                ):
                    continue
                branch = unit.branch.value
                maximum_transport_plane_error = max(
                    maximum_transport_plane_error,
                    abs(float(transport["world_position_m"][2]) - 0.225),
                )
                visible_payload = [
                    geom_id
                    for geom_id in adapter._tray_geom_ids[unit.tray_id]
                    if float(adapter.model.geom_rgba[geom_id, 3]) > 0.05
                ]
                for tray_geom in visible_payload:
                    minimum_table_clearance[branch] = min(
                        minimum_table_clearance[branch],
                        _signed_xy_aabb_clearance_m(
                            adapter,
                            tray_geom,
                            s2b_top,
                        ),
                    )
                    if branch == "ARM3_B":
                        for obstacle_geom in b_south_obstacles:
                            minimum_b_south_cell_clearance = min(
                                minimum_b_south_cell_clearance,
                                _signed_xy_aabb_clearance_m(
                                    adapter,
                                    tray_geom,
                                    obstacle_geom,
                                ),
                            )
                        for arm_geom in arm3_geoms:
                            minimum_arm3_clearance = min(
                                minimum_arm3_clearance,
                                _signed_aabb_clearance_m(
                                    adapter,
                                    tray_geom,
                                    arm_geom,
                                ),
                            )
                checked_samples[branch] += 1
            if runtime.complete and adapter.transport_settled:
                break

        assert all(count >= 10 for count in checked_samples.values())
        # Top-view clearance proves the path goes around S2B instead of
        # exploiting the 0.9 mm tray/worktop render gap.
        assert all(clearance >= 0.050 for clearance in minimum_table_clearance.values())
        assert minimum_b_south_cell_clearance >= 0.049
        # Includes the blue comb frame, fins and press bars against every
        # collision link, rather than checking only the dark tray slab.
        assert minimum_arm3_clearance >= 0.020
        assert maximum_transport_plane_error <= 1.0e-9
    finally:
        adapter.close()


def test_v2_arm1_has_enough_travel_for_the_c_fin_thickness() -> None:
    """The leading C order in a two-order release exercises Arm1's 1.8 mm clamp."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("C", order_id="ARM1_CLAMP_C")
        runtime.submit_order("A", order_id="ARM3_FOLLOW_A")
        observed = False
        for _ in range(16_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            unit = runtime.units["ARM1_CLAMP_C_UNIT_01"]
            state = adapter.robot_motion_snapshot()["arm1"]
            if unit.branch is None or unit.branch.value != "ARM1_A":
                continue
            if ":INSTALL_FIN:" not in str(state["operation"]) or not state["workpiece_held"]:
                continue
            assert _finger_inner_gap_m(adapter, "arm1") == pytest.approx(
                0.0018,
                abs=0.00015,
            )
            observed = True
            break
        assert observed
    finally:
        adapter.close()


def test_v2_arm1_visibly_returns_suction_and_collects_gripper_before_fin_work() -> None:
    """Changing BASE_LOADING -> INSTALL_FIN must be a physical rack sequence."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="TOOLCHANGE_A")
        runtime.submit_order("C", order_id="TOOLCHANGE_C")
        labels: list[str] = []
        toolchange_duration_s = 0.0
        for _ in range(6_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            state = adapter.robot_motion_snapshot()["arm1"]
            label = str(state["target_zh"])
            if "换刀" in label:
                labels.append(label)
                toolchange_duration_s += 0.01
            if any("归还吸盘" in item for item in labels) and any("取用夹爪" in item for item in labels):
                break
        assert any("归还吸盘" in item for item in labels)
        assert any("取用夹爪" in item for item in labels)
        assert toolchange_duration_s >= 0.30
    finally:
        adapter.close()


def test_v2_base_visual_ownership_handoff_has_no_disappearing_frame() -> None:
    """The tray base must appear before the temporary suction proxy vanishes."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="BASE_HANDOFF_A")
        observed_release = False
        for _ in range(2_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            unit = runtime.units["BASE_HANDOFF_A_UNIT_01"]
            state = adapter.robot_motion_snapshot()["arm1"]
            if not state["release_verified"]:
                continue
            assert unit.tray_id is not None
            assert adapter.component_visible(unit.tray_id, "base_plate")
            assert adapter.model.geom("v2_arm1_raw_base_proxy_geom").rgba[3] == pytest.approx(0.0)
            observed_release = True
            break
        assert observed_release
    finally:
        adapter.close()


def test_v2_released_base_remains_visible_until_the_tray_leaves_s1() -> None:
    """A release latch must survive the following BASE_LOADING refresh frames."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="BASE_LATCH_A")
        release_seen = False
        checked_frames = 0
        for _ in range(3_000):
            runtime.tick(0.005)
            adapter.sync(runtime)
            adapter.step_physics(0.005)
            unit = runtime.units["BASE_LATCH_A_UNIT_01"]
            state = adapter.robot_motion_snapshot()["arm1"]
            release_seen |= bool(state["release_verified"])
            if not release_seen:
                continue
            assert unit.tray_id is not None
            assert adapter.component_visible(unit.tray_id, "base_plate")
            checked_frames += 1
            if unit.stage.value != "BASE_LOADING":
                break
        assert release_seen
        assert checked_frames >= 5
    finally:
        adapter.close()


def test_v2_s1_physically_displays_only_one_order_tray_at_a_time() -> None:
    """The next pallet may appear at S1 only after the previous pallet departs."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="S1_FIRST")
        runtime.submit_order("B", order_id="S1_SECOND")
        s1 = np.asarray(runtime.topology.station("S1_BASE_LOADING").world_xyz)
        second_arrived = False
        for _ in range(5_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            visible_at_s1 = [
                tray.tray_id
                for tray in runtime.flow.trays
                if adapter.tray_visible(tray.tray_id)
                and np.linalg.norm(adapter.tray_position(tray.tray_id) - s1) < 0.01
            ]
            assert len(visible_at_s1) <= 1
            second = runtime.units["S1_SECOND_UNIT_01"]
            if second.tray_id is not None and second.tray_id in visible_at_s1:
                first = runtime.units["S1_FIRST_UNIT_01"]
                assert first.tray_id is not None
                assert adapter.physical_owner_snapshot()[first.tray_id] != "S1"
                second_arrived = True
                break
        assert second_arrived
    finally:
        adapter.close()


def test_v2_arm1_toolchange_dock_and_retreat_are_measured_vertical_lines() -> None:
    """Rack insertion/removal must not arc sideways under joint interpolation."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="VERTICAL_TOOL_A")
        runtime.submit_order("C", order_id="VERTICAL_TOOL_C")
        samples: dict[str, list[np.ndarray]] = {
            "归还吸盘到架": [],
            "空法兰退出吸盘": [],
            "取用夹爪并锁定": [],
            "带夹爪平稳撤离": [],
        }
        for _ in range(16_000):
            runtime.tick(0.005)
            adapter.sync(runtime)
            adapter.step_physics(0.005)
            state = adapter.robot_motion_snapshot()["arm1"]
            label = str(state["target_zh"])
            for marker, values in samples.items():
                if marker in label:
                    values.append(np.asarray(state["actual_tcp_position_m"], dtype=float))
            if "带夹爪高位离开工具架" in label and all(len(values) >= 5 for values in samples.values()):
                break
        for marker, values in samples.items():
            assert len(values) >= 5, marker
            positions = np.stack(values)
            assert float(np.ptp(positions[:, 0])) <= 0.001, marker
            assert float(np.ptp(positions[:, 1])) <= 0.001, marker
            z_steps = np.diff(positions[:, 2])
            if marker in {"归还吸盘到架", "取用夹爪并锁定"}:
                assert float(positions[0, 2] - positions[-1, 2]) >= 0.09, marker
                assert float(np.max(z_steps)) <= 0.001, marker
            else:
                assert float(positions[-1, 2] - positions[0, 2]) >= 0.09, marker
                assert float(np.min(z_steps)) >= -0.001, marker
    finally:
        adapter.close()


@pytest.mark.parametrize("fast", (False, True))
def test_v2_arm1_toolchange_clears_the_rack_and_only_moves_sideways_above_it(
    fast: bool,
) -> None:
    """The flange may dock vertically, but must never penetrate rack structure."""

    mujoco = pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=fast)
    adapter = DualLineSceneAdapter(V2_XML)
    rack_geoms = {
        int(adapter.model.geom(name).id)
        for name in (
            "v2_arm1_tool_rack_base",
            "v2_arm1_tool_rack_column",
            "v2_arm1_tool_rack_beam",
            "v2_arm1_tool_rack_left_finger",
            "v2_arm1_tool_rack_right_finger",
        )
    }
    upper_geoms = tuple(
        adapter.model.geom(name)
        for name in (
            "v2_arm1_tool_rack_beam",
            "v2_arm1_tool_rack_left_finger",
            "v2_arm1_tool_rack_right_finger",
        )
    )
    rack_body = adapter.data.body("v2_arm1_tool_rack")
    beam_top_m = float(rack_body.xpos[2] + max(geom.pos[2] + geom.size[2] for geom in upper_geoms))
    safe_plane_z_m = float(adapter.data.site("v2_arm1_tool_rack_safe_plane").xpos[2])
    observed_labels: list[str] = []
    unsafe_contacts: list[tuple[str, str, str]] = []
    high_crossing_z_m: list[float] = []
    try:
        runtime.submit_order("A", order_id="RACK_CLEAR_TOOL_A")
        for _ in range(16_000):
            runtime.tick(0.005)
            adapter.sync(runtime)
            adapter.step_physics(0.005)
            state = adapter.robot_motion_snapshot()["arm1"]
            label = str(state["target_zh"])
            if "换刀" not in label:
                continue
            observed_labels.append(label)
            if "高位横移" in label or "高位离开" in label:
                high_crossing_z_m.append(float(state["actual_tcp_position_m"][2]))
            for contact_index in range(adapter.data.ncon):
                contact = adapter.data.contact[contact_index]
                if contact.geom1 not in rack_geoms and contact.geom2 not in rack_geoms:
                    continue
                other = contact.geom2 if contact.geom1 in rack_geoms else contact.geom1
                other_name = mujoco.mj_id2name(adapter.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                if not other_name.startswith("arm1_fr3_"):
                    continue
                rack = contact.geom1 if contact.geom1 in rack_geoms else contact.geom2
                rack_name = mujoco.mj_id2name(adapter.model, mujoco.mjtObj.mjOBJ_GEOM, rack) or ""
                unsafe_contacts.append((label, rack_name, other_name))
            if "带夹爪高位离开工具架" in label:
                break

        required_phases = (
            "正上方",
            "竖直慢降",
            "竖直慢升",
            "高位离开",
        )
        assert all(any(phase in label for label in observed_labels) for phase in required_phases)
        assert high_crossing_z_m
        assert min(high_crossing_z_m) >= beam_top_m + 0.050
        assert abs(max(high_crossing_z_m) - safe_plane_z_m) <= 0.010
        assert not unsafe_contacts
    finally:
        adapter.close()


def test_v2_fin_final_descent_is_slow_vertical_and_micro_corrected() -> None:
    """The final 105 mm insertion must be visibly slow and strictly vertical."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=False)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="SLOW_INSERT_A")
        started_at: float | None = None
        target_positions: list[np.ndarray] = []
        for _ in range(10_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            unit = runtime.units["SLOW_INSERT_A_UNIT_01"]
            if unit.branch is None:
                continue
            arm_name = "arm1" if unit.branch.value == "ARM1_A" else "arm3"
            state = adapter.robot_motion_snapshot()[arm_name]
            label = str(state["target_zh"])
            if "锁定XY和角度后纯Z下降" in label or "保持角度纯Z向下放置" in label:
                if started_at is None:
                    started_at = float(adapter.data.time)
                target_positions.append(np.asarray(state["target_tcp_position_m"], dtype=float))
            if started_at is not None and state["release_verified"]:
                elapsed = float(adapter.data.time) - started_at
                assert elapsed >= 3.8
                break
        assert started_at is not None
        assert len(target_positions) >= 50
        targets = np.stack(target_positions)
        assert float(np.ptp(targets[:, 0])) <= 1.0e-9
        assert float(np.ptp(targets[:, 1])) <= 1.0e-9
        assert np.all(np.diff(targets[:, 2]) <= 1.0e-9)
        assert float(np.max(np.abs(np.diff(targets[:, 2])))) <= 0.0015 + 1.0e-9
    finally:
        adapter.close()


def test_v2_fin_release_has_no_invisible_ownership_gap() -> None:
    """Proxy-to-tray visual ownership transfers in the release physics tick."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="VISIBLE_HANDOFF_A")
        saw_grasp = False
        saw_release_handoff = False
        last_visible_proxy_position: np.ndarray | None = None
        for _ in range(5_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            unit = runtime.units["VISIBLE_HANDOFF_A_UNIT_01"]
            if unit.tray_id is None:
                continue
            arm_name = "arm1" if unit.branch is None or unit.branch.value == "ARM1_A" else "arm3"
            state = adapter.robot_motion_snapshot()[arm_name]
            if ":INSTALL_FIN:" not in str(state["operation"]):
                if saw_release_handoff and unit.fins_installed >= 1:
                    break
                continue
            proxy_alpha = float(adapter.model.geom(f"v2_{arm_name}_raw_fin_proxy_geom").rgba[3])
            installed_alpha = float(adapter.model.geom(f"{unit.tray_id.lower()}_fin_01").rgba[3])
            saw_grasp |= bool(state["grasp_verified"])
            if saw_grasp:
                assert max(proxy_alpha, installed_alpha) > 0.5
            if proxy_alpha > 0.5:
                last_visible_proxy_position = np.asarray(
                    adapter.data.geom(f"v2_{arm_name}_raw_fin_proxy_geom").xpos,
                    dtype=float,
                ).copy()
            if state["release_verified"] and unit.fins_installed == 0:
                assert installed_alpha > 0.5
                assert last_visible_proxy_position is not None
                installed_position = np.asarray(
                    adapter.data.geom(f"{unit.tray_id.lower()}_fin_01").xpos,
                    dtype=float,
                )
                np.testing.assert_allclose(
                    installed_position,
                    last_visible_proxy_position,
                    atol=0.001,
                )
                saw_release_handoff = True

        assert saw_grasp
        assert saw_release_handoff
        assert runtime.units["VISIBLE_HANDOFF_A_UNIT_01"].fins_installed >= 1
    finally:
        adapter.close()


def test_v2_fin_carry_and_dispensing_follow_v1_pose_locked_cartesian_semantics() -> None:
    """V2 must preserve the two process-motion invariants proven in V1.

    A grasped fin translates without changing its measured 3-D attitude or
    wrist-roll branch.  During deposition, the dispenser TCP follows each
    authored straight line and remains vertical instead of servo-chasing the
    next sparse waypoint.  The assertions observe the live MuJoCo bodies used
    by the viewer, not planner estimates.
    """

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    held_cycles: list[list[tuple[np.ndarray, np.ndarray]]] = []
    current_cycle: list[tuple[np.ndarray, np.ndarray]] = []
    dispense_lines: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    try:
        runtime.submit_order("A", order_id="V1_MOTION_PARITY_A")
        for _ in range(12_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            snapshot = adapter.robot_motion_snapshot()

            held_sample: tuple[np.ndarray, np.ndarray] | None = None
            for arm_name in ("arm1", "arm3"):
                state = snapshot[arm_name]
                if ":INSTALL_FIN:" not in str(state["operation"]) or not state["workpiece_held"]:
                    continue
                proxy = adapter.data.body(f"v2_{arm_name}_raw_fin_proxy")
                held_sample = (
                    np.asarray(state["joint_positions"], dtype=float),
                    np.asarray(proxy.xquat, dtype=float).copy(),
                )
            if held_sample is None:
                if current_cycle:
                    held_cycles.append(current_cycle)
                    current_cycle = []
            else:
                current_cycle.append(held_sample)

            arm2 = snapshot["arm2"]
            label = str(arm2["target_zh"])
            if ":DISPENSING:" in str(arm2["operation"]) and "连续涂覆" in label:
                tcp = adapter.data.site("v2_arm2_dispenser_center_tcp")
                dispense_lines.setdefault(label, []).append(
                    (
                        np.asarray(tcp.xpos, dtype=float).copy(),
                        np.asarray(arm2["target_tcp_position_m"], dtype=float),
                        np.asarray(tcp.xmat, dtype=float).reshape(3, 3)[:, 2].copy(),
                    )
                )
            if runtime.complete:
                break
        if current_cycle:
            held_cycles.append(current_cycle)

        assert runtime.complete
        assert len(held_cycles) == 5
        for cycle in held_cycles:
            assert len(cycle) >= 20
            joints = np.stack([sample[0] for sample in cycle])
            quaternions = [sample[1] for sample in cycle]
            # Arm1 can retain V1's literal q7 lock.  Arm3's longer hybrid
            # head needs smooth redundancy motion, but may never switch IK
            # branches while carrying the fin.
            assert float(np.max(np.abs(np.diff(joints[:, 6])))) <= 0.03
            assert _quaternion_excursion_rad(quaternions) <= np.deg2rad(0.5)

        assert len(dispense_lines) == 5
        for line in dispense_lines.values():
            assert len(line) >= 20
            actual = np.stack([sample[0] for sample in line])
            target = np.stack([sample[1] for sample in line])
            target_span = np.ptp(target, axis=0)
            travel_axis = int(np.argmax(target_span))
            cross_axes = [axis for axis in range(3) if axis != travel_axis]
            # Only evaluate samples that have reached nozzle contact height;
            # longitudinal lag is allowed, lateral/vertical wandering is not.
            contact = np.abs(actual[:, 2] - target[:, 2]) <= 0.002
            assert int(np.count_nonzero(contact)) >= 10
            cross_track = actual[contact][:, cross_axes] - target[contact][:, cross_axes]
            assert float(np.max(np.linalg.norm(cross_track, axis=1))) <= 0.002
            tool_axes = np.stack([sample[2] for sample in line])[contact]
            down_error = np.arccos(np.clip(-tool_axes[:, 2], -1.0, 1.0))
            assert float(np.max(down_error)) <= np.deg2rad(0.2)
    finally:
        adapter.close()


def test_v2_single_nozzle_route_keeps_calibrated_nozzle_tip_height() -> None:
    """The single-nozzle OR branch must reuse the dual route's contact Z."""

    pytest.importorskip("mujoco")
    plan = build_inline_plan(
        preset="A",
        order_id="SINGLE_NOZZLE_HEIGHT",
        quantity=1,
        priority=10,
    )
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    adapter.operation_route_metadata = lambda _resource, _unit_id, kind: (
        {"mode": "SINGLE_TWO_PASS", "capability": "MATERIAL_DISPENSING_SINGLE"}
        if kind == "DISPENSING"
        else {}
    )
    runtime.set_execution_gate(adapter)
    contact_z: list[float] = []
    try:
        runtime.submit_plan(plan)
        unit = runtime.units["SINGLE_NOZZLE_HEIGHT_UNIT_01"]
        for _ in range(12_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            state = adapter.robot_motion_snapshot()["arm2"]
            if ":DISPENSING:" not in str(state["operation"]):
                continue
            if "单喷嘴两遍涂覆" not in str(state["target_zh"]):
                continue
            contact_z.append(float(state["target_tcp_position_m"][2]))
            if len(contact_z) >= 10:
                break

        assert len(contact_z) >= 10
        tray_z = float(adapter.data.body(f"{unit.tray_id.lower()}_carrier").xpos[2])
        expected_z = tray_z + 0.028 + plan.execution_spec.base_thickness + plan.product.nozzle_tip_height_m
        assert min(contact_z) == pytest.approx(expected_z, abs=0.002)
    finally:
        adapter.close()


def test_v2_dispensing_and_fin_installation_are_continuous_without_joint_chatter() -> None:
    """The public physical trace must not contain visible servo reversals.

    This drives the same V2 order path as the viewer.  The limits describe a
    10 ms observation interval and deliberately catch the former per-waypoint
    settle/integral-correction chatter without depending on planner internals.
    """

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    traces: dict[str, list[tuple[int, np.ndarray]]] = {
        "DISPENSING": [],
        "INSTALL_FIN": [],
    }
    try:
        runtime.submit_order("A", order_id="SMOOTH_A")
        for _ in range(12_000):
            runtime.tick(0.01)
            adapter.sync(runtime)
            adapter.step_physics(0.01)
            for state in adapter.robot_motion_snapshot().values():
                operation = str(state["operation"])
                for kind in traces:
                    if f":{kind}:" in operation:
                        traces[kind].append(
                            (
                                int(state["waypoint_index"]),
                                np.asarray(state["joint_positions"], dtype=float),
                            )
                        )
            if traces["DISPENSING"] and traces["INSTALL_FIN"] and runtime.complete:
                break

        for kind, samples in traces.items():
            assert len(samples) >= 40, kind
            joint_steps = np.max(
                np.abs(np.diff(np.stack([sample[1] for sample in samples]), axis=0)),
                axis=1,
            )
            assert float(np.max(joint_steps)) <= 0.035, kind
            assert _direction_reversal_ratio(samples) <= 0.08, kind
    finally:
        adapter.close()


def test_v2_runtime_operations_drive_real_fr3_joint_actuators() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="ROBOT_A")
        runtime.submit_order("B", order_id="ROBOT_B")
        runtime.submit_order("C", order_id="ROBOT_C")
        minimum_tray_clearance = float("inf")
        mechanism_peaks = {
            "lift": 0.0,
            "pusher": 0.0,
            "rear_lift": 0.0,
            "rear_extractor": 0.0,
        }
        fixed_tool_rigid_errors = {"arm2": 0.0, "arm3": 0.0}
        fixed_tool_orientation_errors = {"arm2": 0.0, "arm3": 0.0}
        # Account for the 5 s capture-analysis gate at every inspection while
        # retaining the same 50 ms physical sampling resolution.
        for _ in range(7_000):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            clearance = adapter.visible_tray_clearance_m()
            if clearance is not None:
                minimum_tray_clearance = min(minimum_tray_clearance, clearance)
            for mechanism, value in adapter.furnace_transfer_snapshot().items():
                mechanism_peaks[mechanism] = max(
                    mechanism_peaks[mechanism],
                    value,
                )
            robot_state = adapter.robot_motion_snapshot()
            for arm_name in ("arm2", "arm3"):
                if robot_state[arm_name]["operation"]:
                    fixed_tool_rigid_errors[arm_name] = max(
                        fixed_tool_rigid_errors[arm_name],
                        float(robot_state[arm_name]["tcp_rigid_error_m"]),
                    )
                    fixed_tool_orientation_errors[arm_name] = max(
                        fixed_tool_orientation_errors[arm_name],
                        float(robot_state[arm_name]["tcp_rigid_orientation_error_rad"]),
                    )
            if runtime.complete and adapter.transport_settled:
                break

        snapshot = adapter.robot_motion_snapshot()
        moved = [
            np.linalg.norm(np.asarray(item["joint_positions"]) - HOME_QPOS) for item in snapshot.values()
        ]
        assert sum(distance > 0.02 for distance in moved) >= 2
        assert {"arm1", "arm2", "arm3"} == set(snapshot)
        assert all(item["mode"] == "V1_COMPATIBLE_JOINT_PLAYBACK" for item in snapshot.values())
        assert runtime.complete
        assert adapter.transport_settled
        assert minimum_tray_clearance >= 0.0
        assert mechanism_peaks["lift"] >= 0.13
        assert mechanism_peaks["pusher"] >= 0.50
        assert mechanism_peaks["rear_lift"] >= 0.13
        assert mechanism_peaks["rear_extractor"] >= 0.50
        assert all(error <= 0.003 for error in fixed_tool_rigid_errors.values())
        assert all(error <= np.deg2rad(3.0) for error in fixed_tool_orientation_errors.values())
    finally:
        adapter.close()


def test_arm2_physically_prepositions_while_the_loaded_tray_moves_from_s1_to_s2a() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    observed = None
    try:
        runtime.submit_order("A", order_id="ARM2_PREPOSITION")
        for _ in range(3_000):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            transfers = adapter.transport_snapshot()
            moving_to_s2a = any(
                item["source"] == "S1" and item["target"] == "S2A" for item in transfers.values()
            )
            arm2 = adapter.robot_motion_snapshot()["arm2"]
            if moving_to_s2a and arm2.get("prepositioning"):
                observed = arm2
                assert "ARM2" not in runtime.operations
                assert arm2["preposition_for"] == "DISPENSING"
                assert "安全接近" in str(arm2["target_zh"])
                assert not arm2["workpiece_held"]
                break
    finally:
        adapter.close()

    assert observed is not None


def test_arm1_and_arm3_preposition_during_incoming_tray_motion_without_process_contact() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    observed = {"ARM1": False, "ARM3": False}
    try:
        adapter.sync(runtime)
        runtime.submit_order("A", order_id="MULTI_ARM_PREPOSITION")
        for _ in range(4_000):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            transfers = adapter.transport_snapshot().values()
            routes = {(item["source"], item["target"]) for item in transfers}
            robot_motion = adapter.robot_motion_snapshot()
            if ("EMPTY_BUFFER", "S1") in routes and robot_motion["arm1"].get("prepositioning"):
                observed["ARM1"] = True
                assert "ARM1" not in runtime.operations
                assert robot_motion["arm1"]["preposition_for"] == "BASE_LOADING"
                assert not robot_motion["arm1"]["workpiece_held"]
            if ("S2A", "S2B") in routes and robot_motion["arm3"].get("prepositioning"):
                observed["ARM3"] = True
                assert "ARM3" not in runtime.operations
                assert robot_motion["arm3"]["preposition_for"] == "MATERIAL_INSPECTION"
                assert not robot_motion["arm3"]["workpiece_held"]
            if all(observed.values()):
                break
    finally:
        adapter.close()

    assert observed == {"ARM1": True, "ARM3": True}


def test_arm1_physically_starts_the_gripper_exchange_when_the_base_wave_ends() -> None:
    pytest.importorskip("mujoco")
    runtime = UnifiedV2Runtime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    base_wave_was_complete = False
    observed = None
    try:
        for index, preset in enumerate("ABCAB", start=1):
            runtime.submit_order(preset, order_id=f"TAIL_TOOL_{index}")

        for _ in range(5_000):
            runtime.tick(0.05)
            adapter.sync(runtime.physical_runtime)
            adapter.step_physics(0.05)
            units = tuple(runtime.physical_runtime.units.values())
            base_wave_complete = bool(units) and all(
                unit.stage not in {UnitStage.QUEUED, UnitStage.BASE_LOADING} for unit in units
            )
            if base_wave_complete and not base_wave_was_complete:
                arm1 = adapter.robot_motion_snapshot()["arm1"]
                observed = arm1
                assert arm1["prepositioning"] is True
                assert arm1["preposition_for"] == "INSTALL_FIN"
                assert "换刀" in str(arm1["target_zh"])
                assert arm1["workpiece_held"] is False
                assert "ARM1" not in runtime.physical_runtime.operations
                break
            base_wave_was_complete = base_wave_complete
    finally:
        adapter.close()

    assert observed is not None


def test_six_order_physical_run_has_no_overlap_stall_or_false_grasp_completion() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        for index, preset in enumerate("ABCABC", start=1):
            runtime.submit_order(preset, order_id=f"STRESS_{index}")

        minimum_clearance = float("inf")
        held_seen = False
        grasp_cycles = 0
        previously_held = {"arm1": False, "arm2": False, "arm3": False}
        toolchange_seen = {"gripper": False, "suction": False}
        front_close_lift_positions: list[float] = []
        front_was_open = False
        equality_names = {
            name: int(adapter.model.equality(name).id)
            for name in (
                "v2_arm1_toolchange_parallel_gripper",
                "v2_arm1_toolchange_suction_tool",
                "v2_arm1_rack_parallel_gripper",
                "v2_arm1_rack_suction_tool",
            )
        }

        for _ in range(12_000):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            clearance = adapter.visible_tray_clearance_m()
            if clearance is not None:
                minimum_clearance = min(minimum_clearance, clearance)
            robot_state = adapter.robot_motion_snapshot()
            held_seen |= any(item["workpiece_held"] for item in robot_state.values())
            for arm_name, item in robot_state.items():
                held = bool(item["workpiece_held"])
                if held and not previously_held[arm_name]:
                    grasp_cycles += 1
                previously_held[arm_name] = held
            toolchange_seen["gripper"] |= bool(
                adapter.data.eq_active[equality_names["v2_arm1_toolchange_parallel_gripper"]]
            )
            toolchange_seen["suction"] |= bool(
                adapter.data.eq_active[equality_names["v2_arm1_toolchange_suction_tool"]]
            )
            for tool in ("parallel_gripper", "suction_tool"):
                arm = bool(adapter.data.eq_active[equality_names[f"v2_arm1_toolchange_{tool}"]])
                rack = bool(adapter.data.eq_active[equality_names[f"v2_arm1_rack_{tool}"]])
                assert arm != rack
            front_open = runtime.furnace.state.front_door_open
            if front_was_open and not front_open:
                front_close_lift_positions.append(adapter.furnace_mechanism_position_snapshot()["lift"])
            front_was_open = front_open
            if runtime.complete and adapter.transport_settled:
                break

        assert runtime.complete
        assert adapter.transport_settled
        assert minimum_clearance >= 0.0
        assert held_seen
        assert grasp_cycles >= 38  # 6 bases plus all 32 fins in ABCABC
        assert all(toolchange_seen.values())
        assert front_close_lift_positions
        assert max(front_close_lift_positions) <= 0.01

        runtime.reset()
        adapter.sync(runtime)
        reset_state = adapter.robot_motion_snapshot()["arm1"]["tool_state"]
        assert reset_state["current_tool"] is None
        assert reset_state["arm1_gripper"] == "on_rack"
        assert reset_state["arm1_suction"] == "on_rack"
    finally:
        adapter.close()


@pytest.mark.parametrize(("preset", "fin_count"), (("A", 5), ("B", 4), ("C", 7)))
def test_each_v2_preset_completes_every_fin_pick_insert_open_and_place(
    preset: str,
    fin_count: int,
) -> None:
    """Audit the full visible lifecycle of every fin in an isolated 1x order."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=False)
    adapter = DualLineSceneAdapter(V2_XML)
    order_id = f"FIN_LIFECYCLE_{preset}"
    lifecycle = {index: set() for index in range(1, fin_count + 1)}
    joint_samples: dict[int, list[np.ndarray]] = {index: [] for index in range(1, fin_count + 1)}
    minimum_camera_fin_clearance = float("inf")
    try:
        runtime.submit_order(preset, order_id=order_id)
        unit = runtime.units[f"{order_id}_UNIT_01"]
        for _ in range(12_000):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            if unit.branch is not None:
                arm_name = "arm1" if unit.branch.value == "ARM1_A" else "arm3"
                state = adapter.robot_motion_snapshot()[arm_name]
                if ":INSTALL_FIN:" in str(state["operation"]):
                    index = state["fin_index"]
                    assert isinstance(index, int) and 1 <= index <= fin_count
                    assert not state["failure"]
                    joint_samples[index].append(np.asarray(state["joint_positions"], dtype=float))
                    if state["grasp_verified"] and state["workpiece_held"]:
                        lifecycle[index].add("picked")
                    label = str(state["target_zh"])
                    if "纯Z下降" in label or "纯Z向下放置" in label:
                        lifecycle[index].add("inserted")
                    if state["release_verified"]:
                        lifecycle[index].add("opened")
                        assert unit.tray_id is not None
                        assert adapter.component_visible(unit.tray_id, f"fin_{index:02d}")
                        lifecycle[index].add("placed")
                    if arm_name == "arm3" and state["workpiece_held"]:
                        lens = np.asarray(
                            adapter.data.geom("v2_arm3_camera_lens").xpos,
                            dtype=float,
                        )
                        fin = np.asarray(
                            adapter.data.geom("v2_arm3_raw_fin_proxy_geom").xpos,
                            dtype=float,
                        )
                        clearance = float(np.linalg.norm(lens - fin)) - (0.009 + 0.030)
                        minimum_camera_fin_clearance = min(
                            minimum_camera_fin_clearance,
                            clearance,
                        )
            if runtime.complete and adapter.transport_settled:
                break

        assert runtime.complete
        assert unit.fins_installed == fin_count
        inspections = adapter.inspection_snapshot()
        assert {item["kind"] for item in inspections} == {
            "MATERIAL_INSPECTION",
            "PRE_BRAZE_INSPECTION",
            "POST_BRAZE_INSPECTION",
        }
        assert all(item["aligned"] and item["clear"] for item in inspections)
        assert all(item["analysis_complete"] for item in inspections)
        assert all(float(item["analysis_elapsed_s"]) >= 5.0 for item in inspections)
        expected = {"picked", "inserted", "opened", "placed"}
        assert all(stages == expected for stages in lifecycle.values()), lifecycle
        assert all(joint_samples.values())
        max_joint_step = max(
            float(np.max(np.abs(np.diff(np.stack(samples), axis=0)))) for samples in joint_samples.values()
        )
        assert max_joint_step <= 0.07
        if unit.branch.value == "ARM3_B":
            assert minimum_camera_fin_clearance >= 0.003
    finally:
        adapter.close()
