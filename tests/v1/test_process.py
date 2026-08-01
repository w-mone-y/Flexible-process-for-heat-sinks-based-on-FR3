from __future__ import annotations

import pytest

from brazing_sim.domain import Actor, OrderStage, TaskType
from brazing_sim.process import ActorResult, ProcessCoordinator, TimedTaskActor


def run_to_terminal(coordinator: ProcessCoordinator, limit: float = 40.0):
    now = 0.0
    while now <= limit and not coordinator.terminal:
        coordinator.tick(now)
        now += 0.02
    assert coordinator.product is not None
    assert coordinator.product.terminal, coordinator.snapshot(now)
    return coordinator.product


def test_happy_path_passes() -> None:
    coordinator = ProcessCoordinator(fast=True)
    coordinator.start_order("A", now=0.0, order_id="test-pass")
    product = run_to_terminal(coordinator)
    assert product.stage is OrderStage.PASS
    assert len(product.active_fins) == 5
    assert len(product.active_paths) == 10
    assert all(fin.inserted and fin.board_welded for fin in product.active_fins)
    assert all(path.applied and path.coverage_ratio == 1.0 for path in product.active_paths)
    assert product.fixture.cycle_locked

    load = next(task for task in coordinator.task_history if task.task_type is TaskType.LOAD_FURNACE)
    unload = next(task for task in coordinator.task_history if task.task_type is TaskType.UNLOAD_FURNACE)
    assert load.actor is Actor.CONVEYOR
    assert unload.actor is Actor.CONVEYOR
    assert coordinator.task_history.index(load) < coordinator.task_history.index(unload)


def test_removed_fin_insert_fault_is_rejected_by_v1_coordinator() -> None:
    coordinator = ProcessCoordinator(fast=True)
    with pytest.raises(ValueError, match="unsupported fault type"):
        coordinator.inject_fault("fin_insert", "fin_02")


def test_material_pass_installs_comb_before_fin_assembly() -> None:
    coordinator = ProcessCoordinator(fast=True)
    coordinator.start_order("A", now=0.0, order_id="return-dispenser")
    run_to_terminal(coordinator)

    history = coordinator.task_history
    material_inspection = next(task for task in history if task.task_type is TaskType.MATERIAL_INSPECT)
    configure_comb = next(task for task in history if task.task_type is TaskType.CONFIGURE_COMB)
    first_fin = next(task for task in history if task.task_type is TaskType.INSERT_FIN)

    assert history.index(configure_comb) == history.index(material_inspection) + 1
    assert history.index(first_fin) == history.index(configure_comb) + 1


def test_material_rework_reuses_mounted_dispenser_until_reinspection_passes() -> None:
    coordinator = ProcessCoordinator(fast=True)
    coordinator.inject_fault("brazing_gap", "slot_02_left")
    coordinator.start_order("A", now=0.0, order_id="return-after-rework")
    run_to_terminal(coordinator)

    history = coordinator.task_history
    inspections = [task for task in history if task.task_type is TaskType.MATERIAL_INSPECT]
    reapplications = [task for task in history if task.task_type is TaskType.REAPPLY_MATERIAL]

    assert len(inspections) == 2
    assert len(reapplications) == 1
    assert history.index(inspections[0]) < history.index(reapplications[0])
    assert history.index(reapplications[0]) < history.index(inspections[1])
    configure_comb = next(task for task in history if task.task_type is TaskType.CONFIGURE_COMB)
    first_fin = next(task for task in history if task.task_type is TaskType.INSERT_FIN)
    assert history.index(configure_comb) == history.index(inspections[1]) + 1
    assert history.index(first_fin) == history.index(configure_comb) + 1


def test_material_fault_exists_before_arm3_inspection_starts() -> None:
    coordinator = ProcessCoordinator(fast=True)
    fault = coordinator.inject_fault("brazing_gap", "slot_02_left")
    product = coordinator.start_order("A", now=0.0, order_id="fault-before-material-scan")
    now = 0.0
    while now < 20.0 and not fault.applied:
        coordinator.tick(now)
        now += 0.02

    path = next(path for path in product.active_paths if path.path_id == "slot_02_left")
    assert fault.applied
    assert product.stage is OrderStage.MATERIAL_APPLICATION
    assert path.longest_gap_m > 0.0
    assert not any(task.task_type is TaskType.MATERIAL_INSPECT for task in coordinator.task_history)

    while now < 20.0 and not (
        coordinator.active_task is not None and coordinator.active_task.task_type is TaskType.MATERIAL_INSPECT
    ):
        coordinator.tick(now)
        now += 0.02
    assert coordinator.active_task is not None
    assert path.longest_gap_m > 0.0


def test_fin_pose_fault_exists_before_arm3_geometry_inspection_starts() -> None:
    coordinator = ProcessCoordinator(fast=True)
    fault = coordinator.inject_fault("fin_pose", "fin_02")
    product = coordinator.start_order("A", now=0.0, order_id="fault-before-fin-scan")
    now = 0.0
    while now < 25.0 and not fault.applied:
        coordinator.tick(now)
        now += 0.02

    fin = next(fin for fin in product.active_fins if fin.fin_id == "fin_02")
    assert fault.applied
    assert product.stage is OrderStage.FIN_ASSEMBLY
    assert fin.position_error_m > 0.0
    assert fin.verticality_error_deg > 0.0
    assert not any(task.task_type is TaskType.PRE_INSPECT for task in coordinator.task_history)

    while now < 25.0 and not (
        coordinator.active_task is not None and coordinator.active_task.task_type is TaskType.PRE_INSPECT
    ):
        coordinator.tick(now)
        now += 0.02
    assert coordinator.active_task is not None
    assert fin.position_error_m > 0.0


@pytest.mark.parametrize(
    ("fault_type", "target", "expected_rework"),
    [
        ("fin_pose", "fin_02", "fin"),
        ("fin_pick", "fin_02", "fin"),
        ("brazing_gap", "slot_02_left", "material"),
        ("brazing_deviation", "slot_02_left", "material"),
    ],
)
def test_recoverable_faults_are_reworked(fault_type: str, target: str, expected_rework: str) -> None:
    coordinator = ProcessCoordinator(fast=True)
    coordinator.inject_fault(fault_type, target)
    coordinator.start_order("A", now=0.0)
    product = run_to_terminal(coordinator)
    assert product.stage is OrderStage.PASS
    assert coordinator.kpi.as_dict(40.0)["rework_counts"][expected_rework] >= 1


@pytest.mark.parametrize(
    ("severity", "stage"),
    [("recoverable", OrderStage.REWORK_REQUIRED), ("severe", OrderStage.SCRAPPED)],
)
def test_furnace_fault_disposition(severity: str, stage: OrderStage) -> None:
    coordinator = ProcessCoordinator(fast=True)
    coordinator.inject_fault("furnace_profile", severity=severity)
    coordinator.start_order("A", now=0.0)
    assert run_to_terminal(coordinator).stage is stage


def test_stop_and_reset_release_resources() -> None:
    coordinator = ProcessCoordinator(fast=False)
    coordinator.start_order("A", now=0.0)
    coordinator.tick(0.0)
    assert any(value is not None for value in coordinator.resources.snapshot().values())
    coordinator.stop(0.1)
    assert all(value is None for value in coordinator.resources.snapshot().values())
    assert coordinator.product is not None and coordinator.product.stage is OrderStage.STOPPED
    coordinator.reset()
    assert coordinator.product is None


def test_demo_arm2_segment_precoats_ten_paths_before_any_fin_is_installed() -> None:
    coordinator = ProcessCoordinator(fast=True)
    product = coordinator.start_segment("arm2_motion", now=0.0)
    assert all(not fin.inserted for fin in product.active_fins)
    now = 0.0
    while now < 10.0 and not coordinator.paused:
        coordinator.tick(now)
        now += 0.02
    assert coordinator.paused
    assert product.stage is OrderStage.MATERIAL_APPLICATION
    assert len(product.active_paths) == 10
    assert all(path.applied for path in product.active_paths)

    material_tasks = [task for task in coordinator.task_history if task.task_type is TaskType.APPLY_MATERIAL]
    assert len(material_tasks) == 5
    assert [task.payload["material_sequence_index"] for task in material_tasks] == list(range(5))
    assert [task.payload["reverse_travel"] for task in material_tasks] == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert [task.payload["park_after"] for task in material_tasks] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert [task.payload["continuous_from_previous"] for task in material_tasks] == [
        False,
        True,
        True,
        True,
        True,
    ]


def test_arm1_tool_preparation_runs_concurrently_with_arm2_material() -> None:
    coordinator = ProcessCoordinator(fast=False)
    product = coordinator.start_order("A", now=0.0, order_id="parallel-tool-change")

    now = 0.0
    while now < 2.0:
        coordinator.tick(now)
        snapshot = coordinator.snapshot(now)
        if (
            product.stage is OrderStage.MATERIAL_APPLICATION
            and snapshot["arms"]["arm1"]["task_type"] == "PREPARE_FIN_TOOL"
            and snapshot["arms"]["arm2"]["task_type"] == "APPLY_MATERIAL"
        ):
            break
        now += 0.02

    assert product.stage is OrderStage.MATERIAL_APPLICATION
    assert coordinator.active_task is not None
    assert coordinator.active_task.actor is Actor.ARM2
    assert coordinator.background_tasks["arm1"].task_type is TaskType.PREPARE_FIN_TOOL
    resources = coordinator.resources.snapshot()
    assert resources["arm1_tool_rack"]["owner"] == "arm1"
    assert resources["table2_zone"]["owner"] == "arm2"

    coordinator.pause(now)
    assert coordinator.background_tasks["arm1"].status.value == "READY"
    assert all(value is None for value in coordinator.resources.snapshot().values())
    coordinator.resume(now + 0.1)
    coordinator.tick(now + 0.1)
    assert coordinator.background_tasks["arm1"].status.value == "RUNNING"


def test_demo_fin_assembly_segment_starts_after_material_pass_and_installs_all_fins() -> None:
    coordinator = ProcessCoordinator(fast=True)
    product = coordinator.start_segment("fin_assembly", now=0.0)

    assert product.stage is OrderStage.FIN_ASSEMBLY
    assert product.fixture.base_weld_active
    assert product.fixture.comb_configured and product.fixture.comb_aligned
    assert product.fixture.material_passed
    assert all(path.applied and path.coverage_ratio == 1.0 for path in product.active_paths)
    assert all(not fin.inserted for fin in product.active_fins)
    fin_tasks = [task for task in coordinator.tasks if task.task_type is TaskType.INSERT_FIN]
    assert [task.payload["fin_sequence_index"] for task in fin_tasks] == list(range(len(product.active_fins)))
    assert [task.payload["continuous_from_previous"] for task in fin_tasks] == [
        False,
        *([True] * (len(fin_tasks) - 1)),
    ]
    assert [task.payload["park_after"] for task in fin_tasks] == [
        *([False] * (len(fin_tasks) - 1)),
        True,
    ]

    now = 0.0
    while now < 10.0 and not coordinator.paused:
        coordinator.tick(now)
        now += 0.02

    assert coordinator.paused
    assert product.stage is OrderStage.FIN_ASSEMBLY
    assert all(fin.inserted and fin.temporary_welded for fin in product.active_fins)


def test_furnace_cycle_segment_presses_dwells_ten_seconds_and_returns() -> None:
    coordinator = ProcessCoordinator(fast=True)
    product = coordinator.start_segment("furnace_cycle", now=0.0)

    now = 0.0
    while now < 30.0 and not coordinator.paused:
        coordinator.tick(now)
        now += 0.02

    assert coordinator.paused
    assert product.stage is OrderStage.UNLOADING
    assert product.fixture.press_force_held and product.fixture.cycle_locked
    assert not product.fixture.in_furnace
    assert product.furnace.elapsed_seconds == pytest.approx(10.0)
    assert product.furnace.cycle_started_at is not None

    history = coordinator.task_history
    press = next(task for task in history if task.task_type is TaskType.PRESS_FIXTURE)
    lock = next(task for task in history if task.task_type is TaskType.LOCK_FIXTURE)
    load = next(task for task in history if task.task_type is TaskType.LOAD_FURNACE)
    unload = next(task for task in history if task.task_type is TaskType.UNLOAD_FURNACE)
    assert press.actor is Actor.FIXTURE
    assert load.actor is Actor.CONVEYOR and unload.actor is Actor.CONVEYOR
    assert history.index(press) < history.index(lock) < history.index(load) < history.index(unload)


def test_pause_and_continue_restart_the_current_task() -> None:
    coordinator = ProcessCoordinator(fast=False)
    coordinator.start_segment("inspection_1", now=0.0)
    coordinator.tick(0.0)
    assert coordinator.active_task is not None
    coordinator.pause(0.1)
    assert coordinator.paused and coordinator.active_task is None
    assert all(value is None for value in coordinator.resources.snapshot().values())
    coordinator.resume(0.2)
    coordinator.tick(0.2)
    assert not coordinator.paused and coordinator.active_task is not None


class FailingActor(TimedTaskActor):
    def poll_task(self, now: float) -> ActorResult:
        self.error = "injected actor failure"
        return ActorResult.FAILED


def test_actor_failure_is_terminal_and_emitted_once() -> None:
    coordinator = ProcessCoordinator(actors={"arm1": FailingActor()}, fast=True)
    coordinator.start_order("A", now=0.0)
    coordinator.tick(0.0)
    coordinator.tick(0.1)
    coordinator.tick(0.2)
    assert coordinator.product is not None and coordinator.product.stage is OrderStage.ERROR
    failures = [event for event in coordinator.events if event.kind == "task_failed"]
    assert len(failures) == 1
