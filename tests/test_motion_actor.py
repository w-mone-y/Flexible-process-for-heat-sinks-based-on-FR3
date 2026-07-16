from __future__ import annotations

from pathlib import Path

import numpy as np

from brazing_sim.config import create_product_state
from brazing_sim.motion import PolylineTrajectory, Pose

ROOT = Path(__file__).resolve().parents[1]


def quaternion_distance_rad(left: np.ndarray, right: np.ndarray) -> float:
    similarity = abs(float(np.dot(left, right)))
    return float(2.0 * np.arccos(np.clip(similarity, -1.0, 1.0)))


def test_trajectory_samples_include_waypoints_and_respect_spacing() -> None:
    waypoints = (
        Pose(np.asarray([0.0, 0.0, 0.0])),
        Pose(np.asarray([0.013, 0.0, 0.0])),
        Pose(np.asarray([0.013, 0.017, 0.0])),
    )
    samples = PolylineTrajectory(waypoints, sample_spacing_m=0.01).samples()
    positions = [sample.position for sample in samples]

    assert any(np.allclose(position, waypoints[1].position) for position in positions)
    assert np.allclose(positions[-1], waypoints[-1].position)
    assert max(np.linalg.norm(right - left) for left, right in zip(positions, positions[1:])) <= 0.01


def test_arm1_grasp_welds_activate_and_reset() -> None:
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="grasp-contract")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    try:
        registry = scene.registry
        registry.grasp_base(True)
        registry.grasp_fin("fin_01", True)
        assert scene.data.eq_active[registry.equality_id("arm1_grasp_base")] == 1
        assert scene.data.eq_active[registry.equality_id("arm1_grasp_fin_01")] == 1

        registry.reset_dynamic_welds()
        assert scene.data.eq_active[registry.equality_id("arm1_grasp_base")] == 0
        assert scene.data.eq_active[registry.equality_id("arm1_grasp_fin_01")] == 0
    finally:
        scene.close()


def test_arm1_closes_on_fin_then_releases_it_at_board_slot() -> None:
    from brazing_sim.actors import SceneTaskActor
    from brazing_sim.domain import Actor, TaskSpec, TaskType
    from brazing_sim.process import ActorResult
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="arm1-carry")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    try:
        scene.registry.place_base_on_tray()
        neighbour_names = ("fin_01", "fin_03", "fin_04")
        neighbour_start = {
            name: scene.registry.free_body_pose(name).position.copy() for name in neighbour_names
        }
        actor = SceneTaskActor("arm1", scene, lambda: product)
        actor.start_task(
            TaskSpec(
                "insert-fin-02",
                Actor.ARM1,
                TaskType.INSERT_FIN,
                payload={"fin_id": "fin_02"},
                timeout=60.0,
            ),
            scene.time,
        )
        result = ActorResult.RUNNING
        finger_positions: list[float] = []
        grasp_weld_active: list[int] = []
        previous_fin = scene.registry.free_body_pose("fin_02").position.copy()
        previous_fin_quaternion = scene.registry.free_body_pose("fin_02").quaternion.copy()
        previous_joints = scene.data.qpos[actor.controller.qpos_ids].copy()
        maximum_step_m = 0.0
        maximum_rotation_step_rad = 0.0
        maximum_joint_step_rad = 0.0
        grasp_tcp_offset_m: float | None = None
        carried_tool_roll: list[float] = []
        carried_orientations: list[np.ndarray] = []
        while scene.time < 60.0 and result is ActorResult.RUNNING:
            result = actor.poll_task(scene.time)
            current_fin = scene.registry.free_body_pose("fin_02").position.copy()
            current_fin_quaternion = scene.registry.free_body_pose("fin_02").quaternion.copy()
            maximum_step_m = max(maximum_step_m, float(np.linalg.norm(current_fin - previous_fin)))
            maximum_rotation_step_rad = max(
                maximum_rotation_step_rad,
                quaternion_distance_rad(previous_fin_quaternion, current_fin_quaternion),
            )
            previous_fin = current_fin
            previous_fin_quaternion = current_fin_quaternion
            current_joints = scene.data.qpos[actor.controller.qpos_ids].copy()
            maximum_joint_step_rad = max(
                maximum_joint_step_rad,
                float(np.max(np.abs(current_joints - previous_joints))),
            )
            previous_joints = current_joints
            finger_positions.append(scene.registry.arm1_gripper_closed_fraction())
            grasp_active = int(scene.data.eq_active[scene.registry.equality_id("arm1_grasp_fin_02")])
            grasp_weld_active.append(grasp_active)
            if grasp_active:
                carried_tool_roll.append(float(current_joints[-1]))
                carried_orientations.append(current_fin_quaternion.copy())
            if grasp_active and grasp_tcp_offset_m is None:
                grasp_tcp_offset_m = float(
                    np.linalg.norm(actor.controller.current_tcp_pose().position - current_fin)
                )
            scene.step()

        assert result is ActorResult.SUCCEEDED
        assert scene.time < 45.0
        assert scene.arm1_tools.current_tool == "parallel_gripper"
        assert maximum_step_m < 0.001
        assert maximum_rotation_step_rad < np.deg2rad(1.0)
        # The largest reset occurs only after the fin weld is released; while
        # carrying, joint 7 and the workpiece orientation are exactly locked.
        assert maximum_joint_step_rad < 0.025
        assert grasp_tcp_offset_m is not None and grasp_tcp_offset_m < 0.003
        assert max(finger_positions) > 0.9
        assert any(0.1 < value < 0.9 for value in finger_positions)
        assert finger_positions[-1] < 0.05
        assert max(grasp_weld_active) == 1
        assert carried_tool_roll and np.ptp(carried_tool_roll) < 1e-10
        assert (
            max(quaternion_distance_rad(carried_orientations[0], value) for value in carried_orientations)
            < 1e-10
        )
        assert scene.data.eq_active[scene.registry.equality_id("arm1_grasp_fin_02")] == 0
        assert scene.data.eq_active[scene.registry.equality_id("fin_02_fixture_weld")] == 1
        fin = next(item for item in product.active_fins if item.fin_id == "fin_02")
        target = scene.registry.product_pose().transformed(scene.registry.fin_local_targets[fin.fin_id])
        physical_fin = scene.registry.free_body_pose("fin_02")
        assert np.linalg.norm(physical_fin.position - target.position) < 0.001
        assert quaternion_distance_rad(physical_fin.quaternion, target.quaternion) < np.deg2rad(0.2)
        for name in neighbour_names:
            displacement = np.linalg.norm(
                scene.registry.free_body_pose(name).position - neighbour_start[name]
            )
            assert displacement < 0.001
    finally:
        scene.close()


def test_arm1_suction_transfers_base_then_releases_on_tray() -> None:
    from brazing_sim.actors import SceneTaskActor
    from brazing_sim.domain import Actor, TaskSpec, TaskType
    from brazing_sim.process import ActorResult
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="arm1-suction")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    try:
        actor = SceneTaskActor("arm1", scene, lambda: product)
        actor.start_task(TaskSpec("load-base", Actor.ARM1, TaskType.LOAD_BASE, timeout=60.0), scene.time)
        suction_positions: list[float] = []
        grasp_weld_active: list[int] = []
        previous_base = scene.registry.free_body_pose("base_plate").position.copy()
        previous_base_quaternion = scene.registry.free_body_pose("base_plate").quaternion.copy()
        previous_joints = scene.data.qpos[actor.controller.qpos_ids].copy()
        maximum_step_m = 0.0
        maximum_rotation_step_rad = 0.0
        maximum_joint_step_rad = 0.0
        carried_tool_roll: list[float] = []
        carried_orientations: list[np.ndarray] = []
        result = ActorResult.RUNNING
        while scene.time < 60.0 and result is ActorResult.RUNNING:
            result = actor.poll_task(scene.time)
            current_base = scene.registry.free_body_pose("base_plate").position.copy()
            current_base_quaternion = scene.registry.free_body_pose("base_plate").quaternion.copy()
            maximum_step_m = max(maximum_step_m, float(np.linalg.norm(current_base - previous_base)))
            maximum_rotation_step_rad = max(
                maximum_rotation_step_rad,
                quaternion_distance_rad(previous_base_quaternion, current_base_quaternion),
            )
            previous_base = current_base
            previous_base_quaternion = current_base_quaternion
            current_joints = scene.data.qpos[actor.controller.qpos_ids].copy()
            maximum_joint_step_rad = max(
                maximum_joint_step_rad,
                float(np.max(np.abs(current_joints - previous_joints))),
            )
            previous_joints = current_joints
            suction_positions.append(scene.registry.arm1_suction_fraction())
            grasp_weld_active.append(int(scene.data.eq_active[scene.registry.equality_id("arm1_grasp_base")]))
            if grasp_weld_active[-1]:
                carried_tool_roll.append(float(current_joints[-1]))
                carried_orientations.append(current_base_quaternion.copy())
            scene.step()

        assert result is ActorResult.SUCCEEDED
        assert scene.time < 45.0
        assert scene.arm1_tools.current_tool == "suction_tool"
        assert maximum_step_m < 0.001
        assert maximum_rotation_step_rad < np.deg2rad(0.1)
        assert maximum_joint_step_rad < 0.027
        assert max(suction_positions) > 0.99
        assert any(0.1 < value < 0.9 for value in suction_positions)
        assert suction_positions[-1] < 0.01
        assert max(grasp_weld_active) == 1
        assert carried_tool_roll and np.ptp(carried_tool_roll) < 1e-10
        assert (
            max(quaternion_distance_rad(carried_orientations[0], value) for value in carried_orientations)
            < 1e-10
        )
        assert scene.data.eq_active[scene.registry.equality_id("arm1_grasp_base")] == 0
        assert scene.data.eq_active[scene.registry.equality_id("base_tray_weld")] == 1
        assert quaternion_distance_rad(
            scene.registry.free_body_pose("base_plate").quaternion,
            scene.registry.assembly_base_pose.quaternion,
        ) < np.deg2rad(0.1)
        assert (
            np.linalg.norm(
                scene.registry.free_body_pose("base_plate").position
                - scene.registry.assembly_base_pose.position
            )
            < 0.001
        )
    finally:
        scene.close()


def test_arm1_quick_change_keeps_exactly_one_tool_on_flange() -> None:
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "brazing_line.xml", raw=True)
    try:
        manager = scene.arm1_tools
        assert manager.current_tool is None
        manager.change_tool("suction_tool")
        assert manager.current_tool == "suction_tool"
        assert scene.data.eq_active[manager.arm_weld_ids["suction_tool"]] == 1
        assert scene.data.eq_active[manager.rack_weld_ids["parallel_gripper"]] == 1

        manager.change_tool("parallel_gripper")
        assert manager.current_tool == "parallel_gripper"
        assert scene.data.eq_active[manager.arm_weld_ids["suction_tool"]] == 0
        assert scene.data.eq_active[manager.rack_weld_ids["suction_tool"]] == 1
        assert scene.data.eq_active[manager.arm_weld_ids["parallel_gripper"]] == 1
    finally:
        scene.close()
