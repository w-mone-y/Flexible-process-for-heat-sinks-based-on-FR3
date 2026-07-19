from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brazing_sim.config import create_product_state
from brazing_sim.conveyor import ConveyorPhase, ConveyorTaskActor
from brazing_sim.domain import Actor, FixtureStatus, TaskSpec, TaskType

ROOT = Path(__file__).resolve().parents[1]


def _ready_scene():
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="conveyor-physical")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    scene.registry.place_base_on_tray(snap=True)
    scene.fixture_controller.configure_product(product.spec, product.fixture)
    for fin in product.active_fins:
        scene.registry.place_fin_in_slot(fin.fin_id, snap=True)
        fin.inserted = True
        fin.temporary_welded = True
        product.fixture.temporary_fin_welds.add(fin.fin_id)
    fixture = product.fixture
    fixture.base_weld_active = True
    fixture.material_passed = True
    fixture.fins_passed = True
    scene.fixture_controller.start_press(scene.time, fixture)
    scene.fixture_controller.complete_immediately(fixture)
    scene.fixture_controller.lock(fixture)
    product.furnace.door_fraction = 1.0
    scene.registry.set_furnace_door(1.0, teleport=True)
    return scene, product


def _run_actor(scene, actor: ConveyorTaskActor, limit_s: float = 10.0):
    status = None
    positions = [scene.registry.conveyor_position_m]
    while scene.time < limit_s:
        status = actor.poll_task(scene.time)
        scene.step()
        positions.append(scene.registry.conveyor_position_m)
        if str(status) == "SUCCEEDED":
            break
    return status, np.asarray(positions, dtype=float)


def test_conveyor_rejects_an_unpressed_fixture() -> None:
    from brazing_sim.scene import BrazingScene

    product = create_product_state(order_id="conveyor-gate")
    scene = BrazingScene(ROOT / "brazing_line.xml", order=product, raw=True)
    try:
        product.furnace.door_fraction = 1.0
        actor = ConveyorTaskActor(scene, lambda: product)
        task = TaskSpec("load", Actor.CONVEYOR, TaskType.LOAD_FURNACE)
        with pytest.raises(RuntimeError, match="压紧和夹具锁定"):
            actor.start_task(task, scene.time)
    finally:
        scene.close()


def test_conveyor_moves_the_locked_assembly_out_and_back_without_pose_jump() -> None:
    scene, product = _ready_scene()
    try:
        actor = ConveyorTaskActor(scene, lambda: product)
        tray_start = scene.registry.free_body_pose("assembly_tray")
        fin_start = scene.registry.free_body_pose(product.active_fins[0].fin_id)
        relative_start = fin_start.position - tray_start.position

        load = TaskSpec("load", Actor.CONVEYOR, TaskType.LOAD_FURNACE, timeout=20.0)
        actor.start_task(load, scene.time)
        status, outbound = _run_actor(scene, actor, limit_s=10.0)
        assert str(status) == "SUCCEEDED"
        assert actor.phase is ConveyorPhase.AT_FURNACE
        assert scene.registry.conveyor_position_m == pytest.approx(
            scene.registry.conveyor_travel_m,
            abs=ConveyorTaskActor.POSITION_TOLERANCE_M,
        )
        assert np.all(np.diff(outbound) >= -1.0e-5)
        assert float(np.max(np.abs(np.diff(outbound)))) < 0.001

        tray_furnace = scene.registry.free_body_pose("assembly_tray")
        fin_furnace = scene.registry.free_body_pose(product.active_fins[0].fin_id)
        assert np.linalg.norm((fin_furnace.position - tray_furnace.position) - relative_start) < 5.0e-4
        assert abs(float(np.dot(tray_start.quaternion, tray_furnace.quaternion))) > 0.999999

        product.fixture.in_furnace = True
        product.fixture.status = FixtureStatus.IN_FURNACE
        unload = TaskSpec("unload", Actor.CONVEYOR, TaskType.UNLOAD_FURNACE, timeout=20.0)
        actor.start_task(unload, scene.time)
        status, returning = _run_actor(scene, actor, limit_s=scene.time + 10.0)
        assert str(status) == "SUCCEEDED"
        assert actor.phase is ConveyorPhase.RETURNED
        assert scene.registry.conveyor_position_m == pytest.approx(
            0.0,
            abs=ConveyorTaskActor.POSITION_TOLERANCE_M,
        )
        assert np.all(np.diff(returning) <= 1.0e-5)
        assert float(np.max(np.abs(np.diff(returning)))) < 0.001

        tray_returned = scene.registry.free_body_pose("assembly_tray")
        fin_returned = scene.registry.free_body_pose(product.active_fins[0].fin_id)
        assert np.linalg.norm(tray_returned.position - tray_start.position) < 0.002
        assert np.linalg.norm((fin_returned.position - tray_returned.position) - relative_start) < 5.0e-4
        assert bool(scene.data.eq_active[scene.registry.equality_id("tray_fixture_weld")])
    finally:
        scene.close()


def test_conveyor_stop_and_continue_resumes_from_the_measured_position() -> None:
    scene, product = _ready_scene()
    try:
        actor = ConveyorTaskActor(scene, lambda: product)
        task = TaskSpec("load", Actor.CONVEYOR, TaskType.LOAD_FURNACE, timeout=20.0)
        actor.start_task(task, scene.time)
        for _ in range(700):
            actor.poll_task(scene.time)
            scene.step()
        stopped_at = scene.registry.conveyor_position_m
        assert 0.01 < stopped_at < scene.registry.conveyor_travel_m - 0.01
        actor.cancel()
        assert actor.phase is ConveyorPhase.PAUSED

        actor.start_task(task, scene.time)
        assert actor.state["position_m"] == pytest.approx(stopped_at, abs=0.003)
        status, _positions = _run_actor(scene, actor, limit_s=scene.time + 10.0)
        assert str(status) == "SUCCEEDED"
        assert actor.phase is ConveyorPhase.AT_FURNACE
    finally:
        scene.close()


def test_physical_press_furnace_cycle_and_return_to_table2() -> None:
    from brazing_sim.actors import build_scene_actors
    from brazing_sim.process import ProcessCoordinator
    from brazing_sim.safety import ContactMonitor
    from brazing_sim.scene import BrazingScene

    scene = BrazingScene(ROOT / "brazing_line.xml", order="A", raw=True)
    holder: dict[str, ProcessCoordinator] = {}
    actors = build_scene_actors(scene, lambda: holder["coordinator"].product)
    coordinator = ProcessCoordinator(actors=actors)
    holder["coordinator"] = coordinator
    try:
        product = coordinator.start_segment("furnace_cycle", now=scene.time)
        scene.reset(product, raw=True)
        scene.registry.place_base_on_tray(snap=True)
        scene.fixture_controller.configure_product(product.spec, product.fixture)
        for fin in product.active_fins:
            scene.registry.place_fin_in_slot(fin.fin_id, snap=True)
        for path in product.active_paths:
            scene.registry.set_path_visible(path.path_id, True, coverage=1.0)

        start_pose = scene.registry.free_body_pose("assembly_tray")
        monitor = ContactMonitor(scene.model)
        unexpected = []
        maximum_position = 0.0
        while scene.time < 50.0 and not coordinator.paused and not coordinator.terminal:
            coordinator.tick(scene.time)
            scene.registry.set_furnace_door(product.furnace.door_fraction)
            scene.step()
            maximum_position = max(maximum_position, scene.registry.conveyor_position_m)
            for contact in monitor.unexpected(scene.data):
                pair = {contact.body1, contact.body2}
                press_fin = "fixture_upper_plate" in pair and any(body.startswith("fin_") for body in pair)
                if not press_fin:
                    unexpected.append(contact)

        assert not coordinator.terminal, product.errors
        assert coordinator.paused, (
            f"time={scene.time:.3f} stage={product.stage.value} "
            f"slide={scene.registry.conveyor_position_m:.6f}/"
            f"{float(scene.data.ctrl[scene.registry.handles.conveyor_slide_actuator]):.6f} "
            f"velocity={scene.registry.conveyor_velocity_m_s:.6f} "
            f"door={float(scene.data.qpos[scene.model.jnt_qposadr[scene.registry.handles.furnace_door_joint]]):.6f}/"
            f"{float(scene.data.ctrl[scene.registry.handles.furnace_door_actuator]):.6f} "
            f"phase={product.furnace.phase.value} contacts={unexpected[-3:]}"
        )
        assert product.stage.value == "UNLOADING"
        assert product.furnace.elapsed_seconds == pytest.approx(10.0)
        assert maximum_position >= scene.registry.conveyor_travel_m - 0.002
        assert scene.registry.conveyor_position_m == pytest.approx(0.0, abs=0.002)
        returned_pose = scene.registry.free_body_pose("assembly_tray")
        assert np.linalg.norm(returned_pose.position - start_pose.position) < 0.002
        assert abs(float(np.dot(returned_pose.quaternion, start_pose.quaternion))) > 0.999999
        assert not unexpected, unexpected[:3]
    finally:
        scene.close()
