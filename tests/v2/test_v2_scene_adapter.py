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
from brazing_sim.recovery.fault_models import RecoveryStatus

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
            (0.15, -0.45, 0.225),
            (0.115, -0.475, 0.225),
            (0.115, -1.22, 0.225),
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
        ((1.55, 0.0, 0.225), merge),
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


def test_robot_motion_snapshot_reports_continuous_operation_progress() -> None:
    """The task graph needs segment progress, not a coarse waypoint counter."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="ROBOT_PROGRESS_A")
        active = None
        for _ in range(1_200):
            runtime.tick(0.05)
            adapter.sync(runtime)
            adapter.step_physics(0.05)
            active = next(
                (item for item in adapter.robot_motion_snapshot().values() if item.get("operation")),
                None,
            )
            if active is not None:
                break
        assert active is not None
        assert 0.0 <= float(active["progress"]) <= 1.0
    finally:
        adapter.close()


def test_offline_arm_freezes_its_physical_joint_path_until_recovered() -> None:
    """ARM_UNAVAILABLE must stop both logical time and MuJoCo joint playback."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="PHYSICAL_ARM_HOLD")
        adapter.sync(runtime)
        for _ in range(20):
            adapter.step_physics(0.01)
            adapter.sync(runtime)

        runtime.inject_fault("ARM_UNAVAILABLE", target="ARM1", auto_recover=False)
        adapter.sync(runtime)
        before = adapter.robot_motion_snapshot()["arm1"]
        before_joints = np.asarray(before["joint_positions"], dtype=float)
        before_progress = float(before["progress"])

        for _ in range(20):
            adapter.step_physics(0.01)
            adapter.sync(runtime)

        held = adapter.robot_motion_snapshot()["arm1"]
        np.testing.assert_allclose(held["joint_positions"], before_joints, atol=1.0e-7)
        assert float(held["progress"]) == pytest.approx(before_progress)
        assert "故障隔离" in str(held["target_zh"])

        assert not runtime.recover_resource("ARM1")
        runtime.tick(10.0)
        assert runtime.faults.resource_available("ARM1")
        adapter.sync(runtime)
        for _ in range(10):
            adapter.step_physics(0.01)
        resumed = adapter.robot_motion_snapshot()["arm1"]
        assert float(resumed["progress"]) > before_progress
    finally:
        adapter.close()


def test_fault_hold_freezes_carrier_without_a_resume_teleport() -> None:
    """A timeout pauses tray/lift timing and resumes from the same path point."""

    pytest.importorskip("mujoco")
    adapter = DualLineSceneAdapter(V2_XML, transfer_speed_m_s=0.25)
    tray_id = "V2_TRAY_01"
    try:
        start = adapter.tray_position(tray_id)
        target = start + np.asarray([0.30, 0.0, 0.0])
        adapter._begin_motion(
            tray_id,
            target,
            0.0,
            TrayOwner.S1,
            TrayOwner.S2A,
        )
        adapter._advance_motion(tray_id, 0.20)
        before_hold = adapter.tray_position(tray_id)

        adapter._advance_motion(tray_id, 0.20, paused=True)
        adapter._advance_motion(tray_id, 1.20, paused=True)
        np.testing.assert_allclose(adapter.tray_position(tray_id), before_hold, atol=1.0e-12)

        adapter._advance_motion(tray_id, 1.20, paused=False)
        np.testing.assert_allclose(adapter.tray_position(tray_id), before_hold, atol=1.0e-12)
        adapter._advance_motion(tray_id, 1.30)
        assert np.linalg.norm(adapter.tray_position(tray_id) - before_hold) > 0.0
    finally:
        adapter.close()


def test_local_brazing_rework_preserves_all_existing_beads_on_return() -> None:
    """A local gap repair must never erase the already-applied bead layout."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="LOCAL_BRAZE_REWORK")
        runtime.inject_fault("BRAZING_MISSING", target="path_02")
        unit = runtime.units["LOCAL_BRAZE_REWORK_UNIT_01"]
        observed_rework = False
        for _ in range(5_000):
            runtime.tick(0.02)
            adapter.sync(runtime)
            adapter.step_physics(0.02)
            local_rework = any(
                operation.unit_id == unit.unit_id and operation.kind == "DISPENSING" and operation.recovery
                for operation in runtime.operations.values()
            )
            if local_rework:
                observed_rework = True
                assert unit.tray_id is not None
                visible = [
                    adapter.component_visible(unit.tray_id, f"braze_{index:02d}")
                    for index in range(1, 2 * unit.fin_count + 1)
                ]
                assert all(visible), "局部补涂开始时已有焊道不能整批消失"
                break
        assert observed_rework, "没有观察到托盘返回 S2A 后的局部补涂阶段"
    finally:
        adapter.close()


def test_local_brazing_rework_preserves_beads_during_s2b_to_s2a_return() -> None:
    """Detected material stays on the pallet for every physical return frame."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="LOCAL_BRAZE_RETURN")
        runtime.inject_fault("BRAZING_MISSING", target="path_02")
        unit = runtime.units["LOCAL_BRAZE_RETURN_UNIT_01"]
        return_started = False
        return_frames = 0
        for _ in range(8_000):
            runtime.tick(0.02)
            adapter.sync(runtime)
            adapter.step_physics(0.02)
            return_started |= any(event["type"] == "RECOVERY_RETURN_STARTED" for event in runtime.events)
            if not return_started or unit.tray_id is None:
                continue
            local_repair_running = any(
                operation.unit_id == unit.unit_id and operation.kind == "DISPENSING" and operation.recovery
                for operation in runtime.operations.values()
            )
            if local_repair_running:
                break
            return_frames += 1
            assert all(
                adapter.component_visible(unit.tray_id, f"braze_{index:02d}")
                for index in range(1, 2 * unit.fin_count + 1)
            ), "S2B返回S2A途中不得把已涂焊道投影为零进度"
        assert return_frames > 2, "没有采样到S2B到S2A的实体返程"
    finally:
        adapter.close()


def test_deviated_braze_arm2_removes_before_reapplying_the_target_path() -> None:
    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    labels: list[str] = []
    try:
        runtime.submit_order("A", order_id="DEVIATION_REMOVE_REAPPLY")
        runtime.inject_fault("BRAZING_PATH_DEVIATION", target="path_03")
        for _ in range(8_000):
            runtime.tick(0.02)
            adapter.sync(runtime)
            adapter.step_physics(0.02)
            operation = runtime.operations.get("ARM2")
            if (
                operation is not None
                and operation.recovery
                and operation.recovery_fault_type == "BRAZING_PATH_DEVIATION"
            ):
                label = str(adapter.robot_motion_snapshot()["arm2"]["target_zh"])
                if label and (not labels or labels[-1] != label):
                    labels.append(label)
            if labels and any("重新涂覆" in label for label in labels):
                break
        removal_index = next(index for index, label in enumerate(labels) if "清除" in label)
        reapply_index = next(index for index, label in enumerate(labels) if "重新涂覆" in label)
        assert removal_index < reapply_index
    finally:
        adapter.close()


def test_fin_pose_rework_keeps_press_off_and_uses_arm3_in_slot_reseat() -> None:
    """S4 inspects before pressing; Arm3 reseats the existing defective fin."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="FIN_RESEAT")
        runtime.inject_fault("FIN_POSE", target="fin_04")
        unit = runtime.units["FIN_RESEAT_UNIT_01"]
        saw_s4_inspection = False
        saw_return = False
        rework_started = False
        saw_rework_release = False
        target_local_poses: list[tuple[np.ndarray, np.ndarray]] = []
        recovery_labels: set[str] = set()
        rework_held_positions: list[np.ndarray] = []
        rework_release_position: np.ndarray | None = None
        last_defect_world_position: np.ndarray | None = None
        proxy_takeover_position: np.ndarray | None = None
        for _ in range(12_000):
            runtime.tick(0.02)
            adapter.sync(runtime)
            adapter.step_physics(0.02)
            if unit.tray_id is None:
                continue
            if any(
                operation.unit_id == unit.unit_id and operation.kind == "PRE_BRAZE_INSPECTION"
                for operation in runtime.operations.values()
            ):
                saw_s4_inspection = True
                assert not adapter.component_visible(unit.tray_id, "front_press")
                assert not adapter.component_visible(unit.tray_id, "rear_press")
            saw_return |= any(event["type"] == "RECOVERY_RETURN_STARTED" for event in runtime.events)
            operation = runtime.operations.get("ARM3")
            recovery_running = bool(
                operation is not None
                and operation.unit_id == unit.unit_id
                and operation.kind == "INSTALL_FIN"
                and operation.recovery
            )
            if recovery_running:
                rework_started = True
            if saw_return and not rework_started:
                geom = adapter.model.geom(f"{unit.tray_id.lower()}_fin_04")
                target_local_poses.append(
                    (
                        np.asarray(geom.pos, dtype=float).copy(),
                        np.asarray(geom.quat, dtype=float).copy(),
                    )
                )
                assert all(
                    adapter.component_visible(unit.tray_id, f"fin_{index:02d}")
                    for index in range(1, unit.fin_count + 1)
                )
                last_defect_world_position = np.asarray(
                    adapter.data.geom(f"{unit.tray_id.lower()}_fin_04").xpos,
                    dtype=float,
                ).copy()
            if recovery_running:
                arm3_state = adapter.robot_motion_snapshot()["arm3"]
                recovery_labels.add(str(arm3_state["target_zh"]))
                if proxy_takeover_position is None:
                    proxy_takeover_position = np.asarray(
                        adapter.data.geom("v2_arm3_raw_fin_proxy_geom").xpos,
                        dtype=float,
                    ).copy()
                if arm3_state["workpiece_held"]:
                    rework_held_positions.append(
                        np.asarray(
                            adapter.data.geom("v2_arm3_raw_fin_proxy_geom").xpos,
                            dtype=float,
                        ).copy()
                    )
                if arm3_state["release_verified"]:
                    saw_rework_release = True
                    rework_release_position = np.asarray(
                        adapter.data.geom(f"{unit.tray_id.lower()}_fin_04").xpos,
                        dtype=float,
                    ).copy()
            if runtime.complete and adapter.transport_settled:
                break

        assert saw_s4_inspection
        assert saw_return
        assert len(target_local_poses) > 2
        first_position, first_quaternion = target_local_poses[0]
        for position, quaternion in target_local_poses[1:]:
            np.testing.assert_allclose(position, first_position, atol=1.0e-9)
            np.testing.assert_allclose(quaternion, first_quaternion, atol=1.0e-9)
        assert recovery_labels
        assert saw_rework_release
        assert last_defect_world_position is not None
        assert proxy_takeover_position is not None
        assert rework_held_positions
        assert rework_release_position is not None
        np.testing.assert_allclose(
            proxy_takeover_position,
            last_defect_world_position,
            atol=0.001,
        )
        correction_travel_m = max(
            float(np.linalg.norm(position[:2] - proxy_takeover_position[:2]))
            for position in rework_held_positions
        )
        assert correction_travel_m >= 0.008
        np.testing.assert_allclose(
            rework_held_positions[-1],
            rework_release_position,
            atol=0.0015,
        )
        assert not any("原料位" in label for label in recovery_labels)
        assert any("原槽位" in label or "纠偏" in label for label in recovery_labels)
        returned = next(event for event in runtime.events if event["type"] == "RECOVERY_RETURN_STARTED")
        assert returned["target"] == "INSTALL_B"
        assert runtime.complete
        assert runtime.snapshot()["manual_fault_requests"][0]["status"] == "RECOVERED"
    finally:
        adapter.close()


def test_fin_pose_fault_descends_at_offset_and_hands_off_without_teleport() -> None:
    """The latent pose defect must exist while the held fin is descending."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="FIN_OFFSET_DESCENT")
        runtime.inject_fault("FIN_POSE", target="fin_04")
        unit = runtime.units["FIN_OFFSET_DESCENT_UNIT_01"]
        saw_offset_descent = False
        saw_continuous_handoff = False
        last_held_position: np.ndarray | None = None
        for _ in range(8_000):
            runtime.tick(0.02)
            adapter.sync(runtime)
            adapter.step_physics(0.02)
            if unit.tray_id is None:
                continue
            active = next(
                (
                    (resource.lower(), operation)
                    for resource, operation in runtime.operations.items()
                    if operation.unit_id == unit.unit_id
                    and operation.kind == "INSTALL_FIN"
                    and not operation.recovery
                ),
                None,
            )
            if active is None:
                continue
            arm_name, _operation = active
            state = adapter.robot_motion_snapshot()[arm_name]
            if state["fin_index"] != 4:
                continue
            installed = np.asarray(
                adapter.data.geom(f"{unit.tray_id.lower()}_fin_04").xpos,
                dtype=float,
            ).copy()
            proxy = np.asarray(
                adapter.data.geom(f"v2_{arm_name}_raw_fin_proxy_geom").xpos,
                dtype=float,
            ).copy()
            if state["workpiece_held"]:
                last_held_position = proxy
                if "纯Z下降" in str(state["target_zh"]):
                    saw_offset_descent = True
                    np.testing.assert_allclose(proxy[:2], installed[:2], atol=0.0015)
                    assert proxy[2] >= installed[2] - 0.001
            if state["release_verified"] and adapter.component_visible(unit.tray_id, "fin_04"):
                assert last_held_position is not None
                np.testing.assert_allclose(last_held_position, installed, atol=0.0015)
                saw_continuous_handoff = True
                break

        assert saw_offset_descent
        assert saw_continuous_handoff
    finally:
        adapter.close()


def test_arm3_pick_failure_is_missing_at_s4_then_restored_by_manual_review() -> None:
    """The failed wide-gap grasp remains physical until the ten-second review."""

    pytest.importorskip("mujoco")
    runtime = DualLineRuntime(fast=True)
    adapter = DualLineSceneAdapter(V2_XML)
    try:
        runtime.submit_order("A", order_id="FIN_PICK_REAL")
        runtime.inject_fault("FIN_PICK_FAILED", target="fin_02")
        unit = runtime.units["FIN_PICK_REAL_UNIT_01"]
        saw_empty_carry = False
        saw_missing_at_s4 = False
        saw_manual_review = False
        saw_manual_restore = False
        for _ in range(16_000):
            runtime.tick(0.02)
            adapter.sync(runtime)
            adapter.step_physics(0.02)
            if unit.tray_id is None:
                continue
            operation = runtime.operations.get("ARM3")
            state = adapter.robot_motion_snapshot()["arm3"]
            initial_target_operation = bool(
                operation is not None
                and operation.unit_id == unit.unit_id
                and operation.kind == "INSTALL_FIN"
                and not operation.recovery
                and state["fin_index"] == 2
            )
            if initial_target_operation and "抓取失败" in str(state["target_zh"]):
                saw_empty_carry = True
                assert not state["grasp_verified"]
                assert state["grasp_failed"]
                assert not state["workpiece_held"]
                assert float(state["finger_inner_gap_m"]) > float(state["fin_thickness_m"]) + 0.004
                assert adapter.model.geom("v2_fin_b_raw_fin_02").rgba[3] > 0.5
            if operation is not None and operation.kind == "PRE_BRAZE_INSPECTION" and not saw_missing_at_s4:
                saw_missing_at_s4 = True
                assert not adapter.component_visible(unit.tray_id, "fin_02")
            if runtime.faults.plans:
                plan = next(iter(runtime.faults.plans.values()))
                saw_manual_review |= plan.status is RecoveryStatus.MANUAL_REVIEW
                if plan.status is RecoveryStatus.SUCCEEDED:
                    saw_manual_restore |= adapter.component_visible(unit.tray_id, "fin_02")
            if runtime.complete and adapter.transport_settled:
                break

        assert saw_empty_carry
        assert saw_missing_at_s4
        assert saw_manual_review
        assert saw_manual_restore
        assert not any(event["type"] == "RECOVERY_RETURN_STARTED" for event in runtime.events)
        assert runtime.complete
        assert runtime.snapshot()["manual_fault_requests"][0]["status"] == "RECOVERED"
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
