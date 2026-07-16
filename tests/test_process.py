from __future__ import annotations

import pytest

from brazing_sim.domain import OrderStage
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
    assert len(product.active_fins) == 4
    assert all(fin.inserted and fin.board_welded for fin in product.active_fins)
    assert all(path.applied and path.coverage_ratio == 1.0 for path in product.active_paths)
    assert product.fixture.cycle_locked


@pytest.mark.parametrize(
    ("fault_type", "target", "expected_rework"),
    [
        ("fin_pose", "fin_02", "fin"),
        ("brazing_gap", "fin_02_left", "material"),
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


def test_demo_arm2_segment_prepares_product_and_pauses_after_eight_paths() -> None:
    coordinator = ProcessCoordinator(fast=True)
    product = coordinator.start_segment("arm2_motion", now=0.0)
    assert all(fin.inserted for fin in product.active_fins)
    now = 0.0
    while now < 10.0 and not coordinator.paused:
        coordinator.tick(now)
        now += 0.02
    assert coordinator.paused
    assert product.stage is OrderStage.MATERIAL_APPLICATION
    assert all(path.applied for path in product.active_paths)


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
