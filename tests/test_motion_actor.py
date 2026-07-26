from __future__ import annotations

import math
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


def test_trajectory_validation_restores_live_simulation_state_once_complete() -> None:
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "brazing_line.xml", order="A", raw=True)
    try:
        controller = scene.arms["arm1"]
        start = controller.current_tcp_pose()
        trajectory = PolylineTrajectory(
            (
                start,
                Pose(start.position + np.asarray([0.002, 0.0, 0.0]), start.quaternion),
            ),
            sample_spacing_m=0.001,
        )
        controller.target = Pose(start.position + np.asarray([0.0, 0.0, 0.01]), start.quaternion)
        controller.tcp_target = start
        scene.data.qvel[:] = np.linspace(-0.02, 0.02, scene.model.nv)
        scene.data.ctrl[:] = np.linspace(-0.1, 0.1, scene.model.nu)
        qpos_before = scene.data.qpos.copy()
        qvel_before = scene.data.qvel.copy()
        ctrl_before = scene.data.ctrl.copy()
        target_before = controller.target
        tcp_target_before = controller.tcp_target

        results = controller.validate_trajectory(trajectory)

        assert results and all(result.reachable for result in results)
        assert np.array_equal(scene.data.qpos, qpos_before)
        assert np.array_equal(scene.data.qvel, qvel_before)
        assert np.array_equal(scene.data.ctrl, ctrl_before)
        assert controller.target is target_before
        assert controller.tcp_target is tcp_target_before
    finally:
        scene.close()


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


def test_fin_grasp_constraint_handoff_preserves_the_contact_pose() -> None:
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="fin-grasp-handoff")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    try:
        registry = scene.registry
        raw_pose = registry.free_body_pose("fin_01")
        registry.set_free_body_pose(
            "fin_01",
            Pose(registry.site_pose("arm1_grasp_tcp").position, raw_pose.quaternion),
        )
        registry.set_weld(
            "raw_fin_01_rack_weld",
            True,
            recompute=("raw_material_rack", "fin_01"),
            forward=True,
        )
        before = registry.free_body_pose("fin_01")

        registry.seat_and_grasp_fin("fin_01")

        after = registry.free_body_pose("fin_01")
        assert np.linalg.norm(after.position - before.position) < 5.0e-5
        assert quaternion_distance_rad(after.quaternion, before.quaternion) < 1.0e-12
        assert scene.data.eq_active[registry.equality_id("raw_fin_01_rack_weld")] == 0
        assert scene.data.eq_active[registry.equality_id("arm1_grasp_fin_01")] == 1
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
        task = TaskSpec(
            "insert-fin-02",
            Actor.ARM1,
            TaskType.INSERT_FIN,
            payload={"fin_id": "fin_02"},
            timeout=60.0,
        )
        actor.start_task(task, scene.time)
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
        carried_orientations: list[np.ndarray] = []
        carried_relative_poses: list[Pose] = []
        carried_finger_qpos: list[np.ndarray] = []
        fixture_switched_while_closed = False
        pick_milestone_before_place = False
        place_milestone_seen = False
        while scene.time < 60.0 and result is ActorResult.RUNNING:
            result = actor.poll_task(scene.time)
            pick_milestone_before_place |= bool(task.payload.get("physical_pick_complete")) and not bool(
                task.payload.get("physical_place_complete")
            )
            place_milestone_seen |= bool(task.payload.get("physical_place_complete"))
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
                carried_orientations.append(current_fin_quaternion.copy())
                gripper = scene.registry.free_body_pose("arm1_parallel_gripper")
                fin_pose = scene.registry.free_body_pose("fin_02")
                carried_relative_poses.append(gripper.inverse().transformed(fin_pose))
                carried_finger_qpos.append(
                    np.asarray(
                        [
                            scene.data.qpos[int(scene.model.jnt_qposadr[joint])]
                            for joint in scene.registry.arm1_finger_joints
                        ]
                    )
                )
            if grasp_active and grasp_tcp_offset_m is None:
                grasp_tcp_offset_m = float(
                    np.linalg.norm(actor.controller.current_tcp_pose().position - current_fin)
                )
            fixture_active = int(scene.data.eq_active[scene.registry.equality_id("fin_02_fixture_weld")])
            if fixture_active and not grasp_active and finger_positions[-1] > 0.99:
                fixture_switched_while_closed = True
            scene.step()

        assert result is ActorResult.SUCCEEDED
        assert scene.time < 45.0
        assert scene.arm1_tools.current_tool == "parallel_gripper"
        assert maximum_step_m < 0.001
        assert maximum_rotation_step_rad < np.deg2rad(1.0)
        # The largest reset occurs only after the fin weld is released.
        assert maximum_joint_step_rad < 0.025
        assert grasp_tcp_offset_m is not None and grasp_tcp_offset_m < 1e-6
        assert max(finger_positions) > 0.9
        assert any(0.1 < value < 0.9 for value in finger_positions)
        assert finger_positions[-1] < 0.05
        assert max(grasp_weld_active) == 1
        assert pick_milestone_before_place
        assert place_milestone_seen
        assert carried_finger_qpos
        assert np.max(np.abs(np.asarray(carried_finger_qpos) - 0.020)) < 1e-12
        assert carried_relative_poses
        assert (
            max(
                np.linalg.norm(value.position - carried_relative_poses[0].position)
                for value in carried_relative_poses
            )
            < 1e-9
        )
        assert (
            max(
                quaternion_distance_rad(carried_relative_poses[0].quaternion, value.quaternion)
                for value in carried_relative_poses
            )
            < 1e-9
        )
        assert max(
            quaternion_distance_rad(carried_orientations[0], value) for value in carried_orientations
        ) < np.deg2rad(0.03)
        assert fixture_switched_while_closed
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


def test_arm1_reaches_product_c_fin_07_without_raw_rack_collision() -> None:
    from brazing_sim.actors import SceneTaskActor
    from brazing_sim.config import make_order_spec
    from brazing_sim.domain import Actor, TaskSpec, TaskType
    from brazing_sim.process import ActorResult
    from brazing_sim.scene import BrazingScene

    product = create_product_state(make_order_spec("C"), order_id="arm1-c-fin-07")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    try:
        scene.registry.place_base_on_tray()
        actor = SceneTaskActor("arm1", scene, lambda: product)
        actor.start_task(
            TaskSpec(
                "insert-c-fin-07",
                Actor.ARM1,
                TaskType.INSERT_FIN,
                payload={"fin_id": "fin_07"},
                timeout=60.0,
            ),
            scene.time,
        )
        result = ActorResult.RUNNING
        rack_contact_seen = False
        while scene.time < 60.0 and result is ActorResult.RUNNING:
            result = actor.poll_task(scene.time)
            for index in range(int(scene.data.ncon)):
                contact = scene.data.contact[index]
                body_names = {
                    scene.model.body(int(scene.model.geom_bodyid[int(contact.geom1)])).name,
                    scene.model.body(int(scene.model.geom_bodyid[int(contact.geom2)])).name,
                }
                if body_names == {"arm1_tool_rack", "fin_07"}:
                    rack_contact_seen = True
            scene.step()

        assert result is ActorResult.SUCCEEDED, actor.error
        assert not rack_contact_seen
        assert int(scene.data.eq_active[scene.registry.equality_id("fin_07_fixture_weld")]) == 1
        actual = scene.registry.free_body_pose("fin_07")
        target = scene.registry.product_pose().transformed(scene.registry.fin_local_targets["fin_07"])
        assert np.linalg.norm(actual.position - target.position) < 0.001
        assert quaternion_distance_rad(actual.quaternion, target.quaternion) < np.deg2rad(0.2)
    finally:
        scene.close()


def test_continuous_fin_pick_uses_local_lift_instead_of_remote_park() -> None:
    from brazing_sim.actors import SceneTaskActor
    from brazing_sim.domain import Actor, TaskSpec, TaskType
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="arm1-direct-next-fin")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    try:
        scene.arm1_tools.change_tool("parallel_gripper")
        actor = SceneTaskActor("arm1", scene, lambda: product)
        scene.data.qpos[int(actor.controller.qpos_ids[0])] += 0.20
        scene.mujoco.mj_forward(scene.model, scene.data)
        current = actor.controller.current_tcp_pose()
        task = TaskSpec(
            "insert-next-fin",
            Actor.ARM1,
            TaskType.INSERT_FIN,
            payload={"fin_id": "fin_02", "continuous_from_previous": True},
            timeout=60.0,
        )

        trajectory, _ = actor._task_goal(task)

        local_lift = trajectory.waypoints[1]
        assert np.allclose(local_lift.position[:2], current.position[:2], atol=1e-12)
        assert float(local_lift.position[2]) >= 0.50
        remote_park = actor.park_flange_pose.transformed(actor.controller.tool_transform)
        assert np.linalg.norm(local_lift.position[:2] - remote_park.position[:2]) > 0.05
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
        task = TaskSpec("load-base", Actor.ARM1, TaskType.LOAD_BASE, timeout=60.0)
        actor.start_task(task, scene.time)
        suction_positions: list[float] = []
        grasp_weld_active: list[int] = []
        previous_base = scene.registry.free_body_pose("base_plate").position.copy()
        previous_base_quaternion = scene.registry.free_body_pose("base_plate").quaternion.copy()
        previous_joints = scene.data.qpos[actor.controller.qpos_ids].copy()
        maximum_step_m = 0.0
        maximum_rotation_step_rad = 0.0
        maximum_joint_step_rad = 0.0
        carried_orientations: list[np.ndarray] = []
        carried_relative_poses: list[Pose] = []
        pick_milestone_before_place = False
        place_milestone_seen = False
        result = ActorResult.RUNNING
        while scene.time < 60.0 and result is ActorResult.RUNNING:
            result = actor.poll_task(scene.time)
            pick_milestone_before_place |= bool(task.payload.get("physical_pick_complete")) and not bool(
                task.payload.get("physical_place_complete")
            )
            place_milestone_seen |= bool(task.payload.get("physical_place_complete"))
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
                carried_orientations.append(current_base_quaternion.copy())
                suction_tool = scene.registry.free_body_pose("arm1_suction_tool")
                base_pose = scene.registry.free_body_pose("base_plate")
                carried_relative_poses.append(suction_tool.inverse().transformed(base_pose))
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
        assert pick_milestone_before_place
        assert place_milestone_seen
        assert carried_relative_poses
        assert (
            max(
                np.linalg.norm(value.position - carried_relative_poses[0].position)
                for value in carried_relative_poses
            )
            < 1e-9
        )
        assert (
            max(
                quaternion_distance_rad(carried_relative_poses[0].quaternion, value.quaternion)
                for value in carried_relative_poses
            )
            < 1e-9
        )
        assert max(
            quaternion_distance_rad(carried_orientations[0], value) for value in carried_orientations
        ) < np.deg2rad(0.03)
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


def test_arm1_can_visibly_prepare_gripper_without_entering_table2() -> None:
    from brazing_sim.actors import SceneTaskActor
    from brazing_sim.domain import Actor, TaskSpec, TaskType
    from brazing_sim.process import ActorResult
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="arm1-background-tool-preparation")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    try:
        scene.arm1_tools.change_tool("suction_tool")
        actor = SceneTaskActor("arm1", scene, lambda: product)
        actor.start_task(
            TaskSpec(
                "prepare-fin-tool",
                Actor.ARM1,
                TaskType.PREPARE_FIN_TOOL,
                timeout=60.0,
            ),
            scene.time,
        )

        result = ActorResult.RUNNING
        minimum_table2_planar_clearance = float("inf")
        table2_xy = scene.registry.assembly_base_pose.position[:2]
        while scene.time < 60.0 and result is ActorResult.RUNNING:
            result = actor.poll_task(scene.time)
            tcp_xy = actor.controller.current_tcp_pose().position[:2]
            minimum_table2_planar_clearance = min(
                minimum_table2_planar_clearance,
                float(np.linalg.norm(tcp_xy - table2_xy)),
            )
            scene.step()

        assert result is ActorResult.SUCCEEDED
        assert scene.arm1_tools.current_tool == "parallel_gripper"
        # The complete change remains on the Arm1-side rack/safe-corridor side
        # of Table2 even though its parking pose shares a similar Y value.
        assert minimum_table2_planar_clearance > 0.30
    finally:
        scene.close()


def test_arm2_fixed_dispenser_completes_continuous_material_passes() -> None:
    from brazing_sim.actors import SceneTaskActor
    from brazing_sim.domain import Actor, TaskSpec, TaskType
    from brazing_sim.process import ActorResult
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="arm2-fixed-dispenser")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    try:
        scene.registry.dock_assembly_tray_to_station("s2a", snap=True)
        scene.registry.place_base_on_tray(snap=True)
        weld_id = scene.registry.equality_id("arm2_dispenser_tool_weld")
        assert scene.tools.current_tool == "brazing_dispenser"
        assert scene.tools.available_tools == ("brazing_dispenser",)
        assert int(scene.data.eq_active[weld_id]) == 1

        actor = SceneTaskActor("arm2", scene, lambda: product)
        actor.start_task(
            TaskSpec(
                "arm2-fixed-dispense-1",
                Actor.ARM2,
                TaskType.APPLY_MATERIAL,
                payload={
                    "path_ids": ["slot_01_left", "slot_01_right"],
                    "continuous_from_previous": False,
                    "reverse_travel": False,
                    "park_after": False,
                },
                timeout=60.0,
            ),
            scene.time,
        )
        assert actor._arm2_steps
        result = ActorResult.RUNNING
        max_vertical_error_deg = 0.0
        while scene.time < 45.0 and result is ActorResult.RUNNING:
            result = actor.poll_task(scene.time)
            if (
                scene.tools.current_tool == "brazing_dispenser"
                and actor._material_path_ids
                and actor._material_start is not None
                and actor._material_end is not None
            ):
                tcp_pose = actor.controller.current_tcp_pose()
                material_axis = actor._material_end - actor._material_start
                denominator = max(float(np.dot(material_axis, material_axis)), 1.0e-12)
                along = float(
                    np.clip(
                        np.dot(tcp_pose.position - actor._material_start, material_axis) / denominator,
                        0.0,
                        1.0,
                    )
                )
                closest = actor._material_start + along * material_axis
                if float(np.linalg.norm(tcp_pose.position - closest)) <= 0.025:
                    axis = tcp_pose.rotation[:, 2]
                    max_vertical_error_deg = max(
                        max_vertical_error_deg,
                        math.degrees(math.acos(float(np.clip(np.dot(axis, [0.0, 0.0, -1.0]), -1.0, 1.0)))),
                    )
            scene.step()

        assert result is ActorResult.SUCCEEDED
        assert max_vertical_error_deg <= 0.10
        assert int(scene.data.eq_active[weld_id]) == 1
        assert scene.tools.current_tool == "brazing_dispenser"

        # The next slot starts at the same +X end and uses only a short
        # above-product transition; there is no intermediate return to park.
        second = TaskSpec(
            "arm2-fixed-dispense-2",
            Actor.ARM2,
            TaskType.APPLY_MATERIAL,
            payload={
                "path_ids": ["slot_02_left", "slot_02_right"],
                "continuous_from_previous": True,
                "reverse_travel": True,
                "park_after": True,
            },
            timeout=60.0,
        )
        actor.start_task(second, scene.time)
        assert second.payload["travel_direction"] == "negative_x"
        assert actor.trajectory is not None
        assert max(point.position[2] for point in actor.trajectory.waypoints) < 0.35
        for waypoint in actor.trajectory.waypoints:
            assert np.allclose(waypoint.rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-10)

        result = ActorResult.RUNNING
        deadline = scene.time + 20.0
        while scene.time < deadline and result is ActorResult.RUNNING:
            result = actor.poll_task(scene.time)
            axis = actor.controller.current_tcp_pose().rotation[:, 2]
            max_vertical_error_deg = max(
                max_vertical_error_deg,
                math.degrees(math.acos(float(np.clip(np.dot(axis, [0.0, 0.0, -1.0]), -1.0, 1.0)))),
            )
            scene.step()
        assert result is ActorResult.SUCCEEDED
        assert max_vertical_error_deg <= 0.10
        assert int(scene.data.eq_active[weld_id]) == 1
        scene.reset(product, raw=True)
        assert scene.tools.current_tool == "brazing_dispenser"
        assert int(scene.data.eq_active[weld_id]) == 1
    finally:
        scene.close()


def test_material_marker_grows_from_the_actual_travel_endpoint() -> None:
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "brazing_line.xml", order="A", raw=True)
    try:
        scene.registry.place_base_on_tray(snap=True)
        path_id = "slot_01_left"
        body = scene.data.body(f"{path_id}_brazing_path")
        geom = scene.model.geom(f"{path_id}_brazing_path_geom")

        def relative_x_endpoints(reverse: bool) -> np.ndarray:
            scene.registry.set_path_visible(
                path_id,
                True,
                coverage=0.25,
                reverse=reverse,
            )
            scene.mujoco.mj_forward(scene.model, scene.data)
            center = scene.data.geom_xpos[geom.id]
            axis = scene.data.geom_xmat[geom.id].reshape(3, 3)[:, 2]
            half_length = float(geom.size[1])
            endpoints = np.asarray([center - half_length * axis, center + half_length * axis])
            return np.sort(endpoints[:, 0] - float(body.xpos[0]))

        assert np.allclose(relative_x_endpoints(False), [-0.165, -0.0825], atol=1e-6)
        assert np.allclose(relative_x_endpoints(True), [0.0825, 0.165], atol=1e-6)
    finally:
        scene.close()
