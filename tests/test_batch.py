from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brazing_sim.actors import build_scene_actors
from brazing_sim.batch import BatchCoordinator
from brazing_sim.config import create_batch_state
from brazing_sim.domain import (
    Actor,
    BatchStage,
    FurnacePhase,
    OrderStage,
    RackShelfState,
    TaskSpec,
    TaskType,
    TransferPhase,
    TrayUnitPhase,
)
from brazing_sim.process import ActorResult, ProcessCoordinator
from brazing_sim.scene import BrazingScene

ROOT = Path(__file__).resolve().parents[1]


def _coordinators(*, fast: bool) -> tuple[BrazingScene, ProcessCoordinator, BatchCoordinator]:
    scene = BrazingScene(ROOT / "brazing_line.xml", order="A", raw=True)
    holder: dict[str, BatchCoordinator] = {}
    actors = build_scene_actors(
        scene,
        lambda: holder["batch"].product,
        fast=fast,
    )
    single = ProcessCoordinator(actors=actors, fast=fast)
    batch = BatchCoordinator(scene, single, fast=fast)
    holder["batch"] = batch
    return scene, single, batch


def test_batch_products_are_independent_and_orders_are_fixed() -> None:
    batch = create_batch_state("A", layers=3, batch_id="batch-domain")
    assert [unit.layer_index for unit in batch.units] == [0, 1, 2]
    assert batch.rack.load_order == (0, 1, 2)
    assert batch.rack.unload_order == (2, 1, 0)

    batch.units[0].product.active_fins[0].inserted = True
    batch.units[0].product.active_paths[0].coverage_ratio = 0.5
    assert not batch.units[1].product.active_fins[0].inserted
    assert batch.units[1].product.active_paths[0].coverage_ratio == 0.0

    batch.transition(BatchStage.BUILDING_LAYER, 0.0)
    batch.transition(BatchStage.TRANSFERRING_LAYER, 1.0)
    batch.transition(BatchStage.READY_FOR_BRAZING, 2.0)
    batch.transition(BatchStage.BRAZING, 3.0)
    batch.transition(BatchStage.UNLOADING, 4.0)
    batch.transition(BatchStage.POST_INSPECTION, 5.0)
    batch.transition(BatchStage.COMPLETE, 6.0)
    assert batch.terminal


def test_fast_three_layer_batch_runs_one_shared_cycle_and_passes() -> None:
    scene, _single, coordinator = _coordinators(fast=True)
    try:
        batch = coordinator.start_batch("A", layers=3, now=scene.time)
        deadline = scene.time + 40.0
        while scene.time < deadline and not coordinator.terminal:
            coordinator.tick(scene.time)
            scene.step()

        assert batch.stage is BatchStage.COMPLETE, coordinator.snapshot(scene.time)
        assert [unit.phase for unit in batch.units] == [TrayUnitPhase.INSPECTED] * 3
        assert [unit.product.stage for unit in batch.units] == [OrderStage.PASS] * 3
        assert [shelf.state for shelf in batch.rack.shelves] == [RackShelfState.UNLOADED] * 3
        assert [unit.output_slot for unit in batch.units] == [0, 1, 2]
        assert batch.furnace.elapsed_seconds == pytest.approx(10.0)
        assert coordinator.furnace is not None
        assert coordinator.furnace.state.cycle_started_at is not None
        snapshot = coordinator.snapshot(scene.time)
        assert snapshot["disposition"] == "PASS"
        assert snapshot["rack"]["load_order"] == [0, 1, 2]
        assert snapshot["rack"]["unload_order"] == [2, 1, 0]
        assert all(value is None for value in single_resource_values(coordinator))
    finally:
        scene.close()


def test_next_unit_reset_does_not_release_already_loaded_rack_trays() -> None:
    scene, _single, coordinator = _coordinators(fast=True)
    try:
        batch = coordinator.start_batch("A", layers=3, now=scene.time)
        deadline = scene.time + 20.0
        while scene.time < deadline and batch.active_unit_index < 1:
            coordinator.tick(scene.time)
            scene.step()
        assert batch.active_unit_index == 1
        assert batch.rack.shelves[0].state is RackShelfState.LOCKED
        assert bool(scene.data.eq_active[scene.registry.equality_id("batch_rack_tray_01_weld")])

        while scene.time < deadline and batch.active_unit_index < 2:
            coordinator.tick(scene.time)
            scene.step()
        assert batch.active_unit_index == 2
        assert all(
            bool(scene.data.eq_active[scene.registry.equality_id(f"batch_rack_tray_{index:02d}_weld")])
            for index in (1, 2)
        )
    finally:
        scene.close()


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        ("recoverable", OrderStage.REWORK_REQUIRED),
        ("severe", OrderStage.SCRAPPED),
    ],
)
def test_shared_furnace_fault_classifies_all_three_units(
    severity: str,
    expected: OrderStage,
) -> None:
    scene, single, coordinator = _coordinators(fast=True)
    try:
        single.inject_fault("furnace_profile", severity=severity)
        batch = coordinator.start_batch("A", layers=3, now=scene.time)
        while scene.time < 40.0 and not coordinator.terminal:
            coordinator.tick(scene.time)
            scene.step()
        assert batch.stage is BatchStage.COMPLETE
        assert [unit.product.stage for unit in batch.units] == [expected] * 3
    finally:
        scene.close()


def test_physical_lift_transfer_is_continuous_and_pause_can_resume() -> None:
    scene, _single, coordinator = _coordinators(fast=False)
    try:
        batch = coordinator.start_transfer_demo(now=scene.time)
        positions = [scene.registry.free_body_pose("batch_tray_01").position.copy()]
        observed_phases: set[TransferPhase] = set()
        push_samples: list[tuple[float, float]] = []

        for _ in range(900):
            coordinator.tick(scene.time)
            scene.step()
            positions.append(scene.registry.free_body_pose("batch_tray_01").position.copy())
            observed_phases.add(coordinator.transfer.phase)
        coordinator.pause(scene.time)
        paused_axis = scene.registry.batch_joint_position("batch_outfeed_joint")
        for _ in range(300):
            coordinator.tick(scene.time)
            scene.step()
        assert scene.registry.batch_joint_position("batch_outfeed_joint") == pytest.approx(
            paused_axis,
            abs=0.002,
        )

        coordinator.resume(scene.time)
        deadline = scene.time + 25.0
        while scene.time < deadline and not coordinator.paused:
            coordinator.tick(scene.time)
            scene.step()
            tray_position = scene.registry.free_body_pose("batch_tray_01").position.copy()
            positions.append(tray_position)
            observed_phases.add(coordinator.transfer.phase)
            if coordinator.transfer.phase is TransferPhase.PUSHING:
                push_samples.append(
                    (
                        scene.registry.batch_joint_position("batch_pusher_joint"),
                        float(tray_position[1]),
                    )
                )

        assert coordinator.paused
        assert batch.rack.shelves[0].state is RackShelfState.LOCKED
        assert batch.rack.shelves[0].lock_engaged
        assert batch.units[0].phase is TrayUnitPhase.LOCKED
        assert {
            TransferPhase.ALIGNING,
            TransferPhase.PUSHING,
            TransferPhase.LOCKING,
        }.issubset(observed_phases)
        assert len(push_samples) > 100
        pusher_travel = push_samples[-1][0] - push_samples[0][0]
        tray_travel = push_samples[-1][1] - push_samples[0][1]
        assert pusher_travel > 0.30
        assert tray_travel == pytest.approx(pusher_travel, abs=0.003)
        assert scene.registry.batch_joint_position("batch_rack_lock_joint_0") == pytest.approx(
            0.025,
            abs=0.002,
        )
        steps = np.linalg.norm(np.diff(np.asarray(positions), axis=0), axis=1)
        assert float(np.max(steps)) < 0.001
    finally:
        scene.close()


def test_transfer_demo_opens_physical_door_before_first_tray_moves() -> None:
    scene, _single, coordinator = _coordinators(fast=False)
    try:
        # Match the UI path: the viewer has already been stepping for a while
        # before the user clicks the standalone transfer button.
        scene.step(250)
        started_at = scene.time
        batch = coordinator.start_transfer_demo(now=scene.time)
        assert scene.time == pytest.approx(started_at)
        assert batch.furnace.phase is FurnacePhase.DOOR_OPENING
        assert not coordinator.transfer.operation
        assert batch.units[0].product.fixture.ready_for_transfer
        assert bool(scene.data.eq_active[scene.registry.equality_id("base_tray_weld")])
        assert all(
            bool(scene.data.eq_active[scene.registry.equality_id(f"{fin.fin_id}_fixture_weld")])
            for fin in batch.units[0].product.active_fins
        )
        assert float(scene.model.geom_rgba[scene.registry.geom_id("fixture_front_press_bar"), 3]) > 0.0

        started = False
        for _ in range(2500):
            coordinator.tick(scene.time)
            if coordinator.transfer.operation:
                assert scene.registry.furnace_door_fraction >= 0.98
                assert batch.furnace.door_open
                started = True
                break
            assert scene.registry.batch_joint_position("batch_outfeed_joint") == pytest.approx(
                0.0,
                abs=0.001,
            )
            assert scene.registry.batch_joint_position("batch_lift_joint") == pytest.approx(
                0.0,
                abs=0.001,
            )
            assert scene.registry.batch_joint_position("batch_pusher_joint") == pytest.approx(
                0.0,
                abs=0.001,
            )
            scene.step()

        assert started, "tray transfer did not start after the furnace door opened"
        assert not coordinator.transfer.prefetch_active_for(1)
        assert scene.registry.batch_joint_position("batch_tray_02_index_joint") == pytest.approx(
            0.0,
            abs=0.001,
        )
    finally:
        scene.close()


def test_next_empty_tray_indexes_while_previous_tray_enters_rack() -> None:
    scene, _single, coordinator = _coordinators(fast=False)
    try:
        coordinator.start_transfer_demo(now=scene.time)
        # Exercise the production overlap without waiting for the full robot
        # build that normally precedes this already-prepared transfer fixture.
        coordinator.transfer_demo = False
        observed_overlap = False
        deadline = scene.time + 25.0
        while scene.time < deadline:
            coordinator.tick(scene.time)
            if coordinator.transfer.operation == "load" and coordinator.transfer.prefetch_active_for(1):
                observed_overlap = True
                assert coordinator.transfer.phase in {
                    TransferPhase.LIFTING,
                    TransferPhase.PUSHING,
                    TransferPhase.RETRACTING,
                    TransferPhase.LOWERING,
                }
                break
            scene.step()

        assert observed_overlap
        resources = coordinator.single.resources.snapshot()
        assert resources["tray_indexer"]["owner"] == "batch_prefetch"
        assert resources["table2_zone"]["owner"] == "batch_prefetch"

        coordinator.pause(scene.time)
        paused_index = scene.registry.batch_joint_position("batch_tray_02_index_joint")
        paused_lift = scene.registry.batch_joint_position("batch_lift_joint")
        for _ in range(250):
            coordinator.tick(scene.time)
            scene.step()
        assert scene.registry.batch_joint_position("batch_tray_02_index_joint") == pytest.approx(
            paused_index,
            abs=0.002,
        )
        assert scene.registry.batch_joint_position("batch_lift_joint") == pytest.approx(
            paused_lift,
            abs=0.002,
        )
        coordinator.resume(scene.time)
        assert not coordinator.paused
    finally:
        coordinator.reset()
        scene.close()


def test_top_shelf_round_trip_lowers_before_finished_buffer_output() -> None:
    scene, _single, coordinator = _coordinators(fast=False)
    try:
        batch = coordinator.start_batch("A", layers=3, now=scene.time)
        coordinator.single.tasks.clear()
        coordinator.single.active_task = None
        coordinator.single.paused = True
        unit = batch.units[2]
        unit.phase = TrayUnitPhase.READY_FOR_TRANSFER

        coordinator.transfer.start_index(2, scene.time)
        result = None
        index_deadline = scene.time + 15.0
        while scene.time < index_deadline:
            result = coordinator.transfer.poll(scene.time)
            scene.step()
            if str(result) == "SUCCEEDED":
                break
        assert str(result) == "SUCCEEDED"

        coordinator.transfer.start_load(2, scene.time)
        load_deadline = scene.time + 30.0
        while scene.time < load_deadline:
            result = coordinator.transfer.poll(scene.time)
            scene.step()
            if str(result) == "SUCCEEDED":
                break
        assert str(result) == "SUCCEEDED"
        assert batch.rack.shelves[2].state is RackShelfState.LOCKED

        unit.phase = TrayUnitPhase.BRAZED
        batch.rack.shelves[2].state = RackShelfState.BRAZED
        coordinator.transfer.start_unload(2, scene.time)
        positions = [scene.registry.free_body_pose("batch_tray_03").position.copy()]
        deadline = scene.time + 45.0
        while scene.time < deadline:
            result = coordinator.transfer.poll(scene.time)
            scene.step()
            positions.append(scene.registry.free_body_pose("batch_tray_03").position.copy())
            if str(result) == "SUCCEEDED":
                break

        assert str(result) == "SUCCEEDED"
        output = scene.registry.free_body_pose("batch_tray_03")
        target = scene.registry.site_pose("batch_output_slot_03_site")
        assert np.linalg.norm(output.position - target.position) < 0.002
        steps = np.linalg.norm(np.diff(np.asarray(positions), axis=0), axis=1)
        assert float(np.max(steps)) < 0.001
    finally:
        scene.close()


def test_arm3_reaches_all_three_finished_product_buffers() -> None:
    scene, single, coordinator = _coordinators(fast=False)
    try:
        batch = coordinator.start_batch("A", layers=3, now=scene.time)
        single.tasks.clear()
        single.active_task = None
        actor = single.actors[Actor.ARM3.value]
        for index in (2, 1, 0):
            batch.active_unit_index = index
            target = scene.registry.site_pose(f"batch_output_slot_{index + 1:02d}_site")
            task = TaskSpec(
                task_id=f"physical_batch_scan_{index + 1:02d}",
                actor=Actor.ARM3,
                task_type=TaskType.POST_INSPECT,
                payload={
                    "world_position": target.position.tolist(),
                    "top_clearance_m": 0.22,
                    "side_clearance_m": 0.12,
                    "top_yaw_rad": 0.0,
                    "side_yaw_rad": 0.0,
                    "park_after": True,
                },
                timeout=45.0,
            )
            actor.start_task(task, scene.time)
            result = ActorResult.RUNNING
            deadline = scene.time + 45.0
            while scene.time < deadline and result is ActorResult.RUNNING:
                result = actor.poll_task(scene.time)
                scene.step()
            assert result is ActorResult.SUCCEEDED, getattr(actor, "error", "")
    finally:
        scene.close()


def test_unit_failure_stops_the_whole_batch_before_rack_loading() -> None:
    scene, single, coordinator = _coordinators(fast=True)
    try:
        batch = coordinator.start_batch("A", layers=3, now=scene.time)
        assert single.product is not None
        single.product.stage = OrderStage.MANUAL_REVIEW
        coordinator.tick(scene.time)
        assert batch.stage is BatchStage.MANUAL_REVIEW
        assert batch.active_unit.phase is TrayUnitPhase.MANUAL_REVIEW
        assert all(shelf.state is RackShelfState.EMPTY for shelf in batch.rack.shelves)
        assert not coordinator.transfer.operation
    finally:
        scene.close()


def test_reset_homes_axes_releases_rack_and_restores_safe_door() -> None:
    scene, _single, coordinator = _coordinators(fast=True)
    try:
        coordinator.start_transfer_demo(now=scene.time)
        while not coordinator.paused:
            coordinator.tick(scene.time)
            scene.step()
        scene.registry.set_furnace_door(1.0, teleport=True)

        coordinator.reset()

        assert coordinator.batch is None
        for joint in (
            "batch_outfeed_joint",
            "batch_lift_joint",
            "batch_pusher_joint",
            "batch_output_joint",
        ):
            assert scene.registry.batch_joint_position(joint) == pytest.approx(0.0, abs=1e-9)
        for index in range(1, 4):
            assert not bool(
                scene.data.eq_active[scene.registry.equality_id(f"batch_rack_tray_{index:02d}_weld")]
            )
        door_joint = int(scene.model.joint("furnace_door_joint").id)
        door_qpos = int(scene.model.jnt_qposadr[door_joint])
        assert float(scene.data.qpos[door_qpos]) == pytest.approx(0.0, abs=1e-9)
        assert all(value is None for value in single_resource_values(coordinator))
    finally:
        scene.close()


def single_resource_values(coordinator: BatchCoordinator) -> list[object]:
    return list(coordinator.single.resources.snapshot().values())
