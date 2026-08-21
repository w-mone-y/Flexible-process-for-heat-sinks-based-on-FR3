from __future__ import annotations

from brazing_sim.dual_line import UnifiedV2Runtime
from brazing_sim.dual_line.presentation import V2StatePresenter
from brazing_sim.dual_line.tray_flow import TrayOwner
from brazing_sim.flexible import build_inline_plan
from brazing_sim.planning import TaskType


class _InstantPhysicalGate:
    """Measured gate stub: scheduling still crosses the V2 physical bridge."""

    def tray_ready(self, _tray_id: str, _owner: TrayOwner) -> bool:
        return True

    def owner_available(self, _owner: TrayOwner) -> bool:
        return True

    def operation_complete(self, _resource: str, _unit_id: str, _kind: str) -> bool:
        return True

    def operation_start_allowed(self, _resource: str, _unit_id: str, _kind: str) -> bool:
        return True

    def operation_milestone(
        self,
        _resource: str,
        _unit_id: str,
        _kind: str,
        _milestone: str,
    ) -> bool:
        return True


def _plan(order_id: str, route_strategy: str, *, quantity: int = 1):
    return build_inline_plan(
        preset="A",
        order_id=order_id,
        quantity=quantity,
        priority=20,
        route_strategy=route_strategy,
    )


def _run(runtime: UnifiedV2Runtime, *, limit: int = 12_000) -> dict[str, object]:
    for _index in range(limit):
        runtime.tick(0.02)
        if runtime.complete:
            return runtime.snapshot()
    raise AssertionError("unified V2 route did not complete")


def test_v2_dispatch_consumes_configured_or_branch_before_physical_start() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.set_execution_gate(_InstantPhysicalGate())
    runtime.submit_plan(_plan("ROUTE_OR_DISPATCH", "STANDARD"))

    dispense = next(
        task
        for task in runtime.manufacturing_runtime.graph
        if task.task_type is TaskType.DISPENSE_BRAZING
    )
    assert {choice["mode"] for choice in dispense.payload["capability_choices"]} == {
        "DUAL_NOZZLE",
        "SINGLE_TWO_PASS",
    }

    for _index in range(2_000):
        runtime.tick(0.02)
        if any(
            event.get("type") == "OPERATION_STARTED" and event.get("kind") == "DISPENSING"
            for event in runtime.physical_runtime.events
        ):
            break
    else:
        raise AssertionError("V2 did not start the configured dispensing operation")

    started = next(
        event
        for event in runtime.physical_runtime.events
        if event.get("type") == "OPERATION_STARTED" and event.get("kind") == "DISPENSING"
    )
    assert dispense.assigned_resource == "ARM2"
    assert dispense.payload["selected_alternative"]["mode"] == "DUAL_NOZZLE"
    assert started["capability"] == "MATERIAL_DISPENSING_DUAL"
    assert started["route_mode"] == "DUAL_NOZZLE"
    assert runtime.snapshot()["route_execution"]["active_operations"][0]["route_mode"] == "DUAL_NOZZLE"


def test_v2_single_nozzle_or_branch_reaches_the_physical_skill() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.set_execution_gate(_InstantPhysicalGate())
    runtime.submit_plan(_plan("ROUTE_OR_SINGLE", "STANDARD"))
    dispense = next(
        task
        for task in runtime.manufacturing_runtime.graph
        if task.task_type is TaskType.DISPENSE_BRAZING
    )
    dispense.payload["capability_choices"] = [
        choice
        for choice in dispense.payload["capability_choices"]
        if choice["mode"] == "SINGLE_TWO_PASS"
    ]

    for _index in range(2_000):
        runtime.tick(0.02)
        if any(
            event.get("type") == "OPERATION_STARTED" and event.get("kind") == "DISPENSING"
            for event in runtime.physical_runtime.events
        ):
            break
    else:
        raise AssertionError("V2 did not start the single-nozzle dispensing operation")

    started = next(
        event
        for event in runtime.physical_runtime.events
        if event.get("type") == "OPERATION_STARTED" and event.get("kind") == "DISPENSING"
    )
    operation = runtime.physical_runtime.operations["ARM2"]
    assert dispense.payload["selected_alternative"]["mode"] == "SINGLE_TWO_PASS"
    assert started["capability"] == "MATERIAL_DISPENSING_SINGLE"
    assert started["route_mode"] == "SINGLE_TWO_PASS"
    assert operation.route_mode == "SINGLE_TWO_PASS"
    assert operation.duration_s > runtime.physical_runtime.durations.dispensing


def test_v2_high_reliability_route_is_visible_from_physical_events_to_ui() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.set_execution_gate(_InstantPhysicalGate())
    runtime.submit_plan(_plan("ROUTE_HIGH_UI", "HIGH_RELIABILITY"))
    snapshot = _run(runtime)

    closeups = [
        event
        for event in snapshot["events"]
        if event.get("type") == "OPERATION_STARTED"
        and event.get("route_phase") == "S3B_CLOSEUP"
    ]
    assert {(event["kind"], event["resource"]) for event in closeups} == {
        ("MATERIAL_INSPECTION", "ARM3"),
        ("PRE_BRAZE_INSPECTION", "ARM3"),
    }
    assert all(event["route_strategy"] == "HIGH_RELIABILITY" for event in closeups)

    state = V2StatePresenter().present(snapshot, simulation_speed=1.0, actual_rtf=1.0)
    assert state["orders"][0]["route_strategy"] == "HIGH_RELIABILITY"
    assert state["route_execution"]["orders"][0]["route_strategy"] == "HIGH_RELIABILITY"
    review_tasks = [
        task
        for task in state["tasks"]
        if task["task_type"] in {"REVIEW_BRAZING_CLOSEUP", "REVIEW_FINS_CLOSEUP"}
    ]
    assert len(review_tasks) == 2
    assert all(task["status"] == "SUCCEEDED" and "S3B" in task["display_detail_zh"] for task in review_tasks)
    install_tasks = [
        task
        for task in runtime.manufacturing_runtime.graph
        if task.task_type is TaskType.INSTALL_FIN
    ]
    assert all(task.payload["non_preemptive"] for task in install_tasks)
    assert snapshot["camera_coordination"]["rules"]["non_preemptive_fin"] is True


def test_v2_first_article_only_routes_the_first_unit_through_closeups() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.set_execution_gate(_InstantPhysicalGate())
    runtime.submit_plan(_plan("ROUTE_FIRST_ARTICLE", "FIRST_ARTICLE", quantity=2))
    snapshot = _run(runtime)

    closeup_units = {
        event["unit_id"]
        for event in snapshot["events"]
        if event.get("type") == "OPERATION_STARTED" and event.get("route_phase") == "S3B_CLOSEUP"
    }
    assert closeup_units == {"ROUTE_FIRST_ARTICLE_UNIT_01"}
    assert all(unit["stage"] == "COMPLETE" for unit in snapshot["units"])
