from __future__ import annotations

import pytest

from brazing_sim.experiments import MetricsCollector, compare_experiments
from brazing_sim.flexible import build_inline_plan, build_preset_plan
from brazing_sim.manufacturing_runtime import ManufacturingRuntime, OrderRunStatus
from brazing_sim.recovery import FaultType, RecoveryStatus


def run(runtime: ManufacturingRuntime, *, limit: int = 10000, step: float = 0.25) -> None:
    for index in range(limit):
        runtime.tick(index * step)
        if runtime.terminal:
            return
    raise AssertionError("runtime did not terminate")


def test_dynamic_runtime_completes_order_and_records_real_metrics() -> None:
    runtime = ManufacturingRuntime(scheduler_mode="dynamic")
    metrics = MetricsCollector()
    runtime.events.subscribe(None, metrics.handle_event)
    runtime.submit_plan(build_preset_plan("B", quantity=1))
    run(runtime)
    assert next(iter(runtime.orders.values())).status is OrderRunStatus.COMPLETED
    values = metrics.calculate(runtime)
    assert values["completed_units"] == 1
    assert values["task_succeeded"] > 0
    assert 0.0 <= values["average_robot_utilization"] <= 1.0


def test_runtime_performs_one_authoritative_ready_refresh_per_idle_tick(monkeypatch) -> None:
    runtime = ManufacturingRuntime()
    runtime.submit_plan(build_preset_plan("A", quantity=1))
    runtime.tick(0.0)
    calls = 0
    original = runtime._refresh_ready

    def counted_refresh(now: float):
        nonlocal calls
        calls += 1
        return original(now)

    monkeypatch.setattr(runtime, "_refresh_ready", counted_refresh)
    runtime.tick(0.001)

    assert calls == 1


def test_brazing_fault_inserts_idempotent_rework_chain_and_continues() -> None:
    runtime = ManufacturingRuntime()
    runtime.submit_plan(build_preset_plan("A", quantity=1))
    injected = False
    for index in range(10000):
        runtime.tick(index * 0.25)
        dispense = next(task for task in runtime.graph if task.task_type.value == "DISPENSE_BRAZING")
        if not injected and dispense.status.value == "SUCCEEDED":
            inspect = next(task for task in runtime.graph if task.task_type.value == "INSPECT_BRAZING")
            fault = runtime.inject_fault(
                FaultType.BRAZING_MISSING,
                source="ARM3",
                related_task_id=inspect.task_id,
                now=index * 0.25,
                details={"path_ids": ["slot_01_left"]},
            )
            first = runtime.recovery.plan(fault, runtime.graph, index * 0.25)
            second = runtime.recovery.plan(fault, runtime.graph, index * 0.25)
            assert first is second
            injected = True
        if runtime.terminal:
            break
    assert injected and runtime.terminal
    plan = next(iter(runtime.recovery.plans.values()))
    assert plan.status is RecoveryStatus.SUCCEEDED
    assert next(iter(runtime.orders.values())).status is OrderRunStatus.COMPLETED


def test_manual_fault_waits_for_matching_task_then_runs_recovery() -> None:
    runtime = ManufacturingRuntime()
    runtime.submit_plan(build_preset_plan("A", quantity=1))
    request = runtime.arm_manual_fault(
        FaultType.BRAZING_MISSING,
        target="slot_01_left",
        details={"path_ids": ["slot_01_left"], "manual": True},
    )
    assert request.status == "ARMED"

    run(runtime)

    assert request.status == "FIRED"
    assert request.fault_id in runtime.faults
    assert runtime.faults[request.fault_id].recovered
    assert next(iter(runtime.recovery.plans.values())).status is RecoveryStatus.SUCCEEDED


def test_manual_arm_fault_waits_for_resource_use_and_auto_recovers() -> None:
    runtime = ManufacturingRuntime()
    runtime.submit_plan(build_preset_plan("A", quantity=1))
    request = runtime.arm_manual_fault(
        FaultType.ARM_UNAVAILABLE,
        target="ARM1",
        source="ARM1",
        details={"resource_id": "ARM1", "duration": 1.0, "manual": True},
    )

    assert request.status == "ARMED"
    runtime.tick(0.0)
    runtime.tick(0.25)
    assert request.status == "FIRED"
    assert runtime.resources.states["ARM1"].status.value == "FAULTED"
    runtime.tick(1.25)
    assert runtime.resources.states["ARM1"].status.value != "FAULTED"
    assert runtime.faults[request.fault_id].recovered


def test_arm_unavailable_does_not_destroy_other_orders_and_recovers() -> None:
    runtime = ManufacturingRuntime()
    runtime.submit_plan(build_inline_plan(preset="A", order_id="O1", quantity=1, priority=10))
    runtime.submit_plan(build_inline_plan(preset="B", order_id="O2", quantity=1, priority=5))
    runtime.inject_fault(
        FaultType.ARM_UNAVAILABLE,
        source="ARM3",
        now=2.0,
        details={"resource_id": "ARM3", "duration": 2.0},
    )
    run(runtime)
    assert all(entry.status is OrderRunStatus.COMPLETED for entry in runtime.orders.values())
    assert next(iter(runtime.faults.values())).recovered


def test_rack_layer_fault_reassigns_only_pending_unit() -> None:
    runtime = ManufacturingRuntime()
    runtime.submit_plan(build_preset_plan("C", quantity=2))
    fault = runtime.inject_fault(
        FaultType.RACK_LAYER_UNAVAILABLE,
        source="rack",
        now=0.0,
        details={"layer_id": 0},
    )
    assert fault.recovered
    changed = runtime.replanner.history[-1].changed_tasks
    assert changed
    assert len({runtime.graph.get(task_id).unit_id for task_id in changed}) == 1


@pytest.mark.parametrize(
    ("task_type", "fault_type", "details"),
    [
        ("INSPECT_FINS", FaultType.FIN_GEOMETRY_FAILED, {"fin_id": "fin_01"}),
        ("MOVE_ELEVATOR", FaultType.ELEVATOR_TIMEOUT, {}),
        ("RUN_FURNACE", FaultType.FURNACE_DOOR_INTERLOCK, {}),
    ],
)
def test_mandatory_recovery_branches_resume_the_order(task_type, fault_type, details) -> None:
    runtime = ManufacturingRuntime()
    runtime.submit_plan(build_preset_plan("A", quantity=1))
    injected = False
    for index in range(10000):
        runtime.tick(index * 0.25)
        target = next(task for task in runtime.graph if task.task_type.value == task_type)
        if not injected and target.status.value == "RUNNING":
            runtime.inject_fault(
                fault_type,
                source=target.assigned_resource or "runtime",
                related_task_id=target.task_id,
                now=index * 0.25,
                details=details,
            )
            injected = True
        if runtime.terminal:
            break
    assert injected and runtime.terminal
    assert next(iter(runtime.recovery.plans.values())).status is RecoveryStatus.SUCCEEDED
    assert next(iter(runtime.orders.values())).status is OrderRunStatus.COMPLETED


def test_experiment_comparison_uses_real_values() -> None:
    result = compare_experiments(
        {"makespan": 100, "average_robot_utilization": 0.5},
        {"makespan": 80, "average_robot_utilization": 0.6},
    )
    assert result["makespan"]["percent_change"] == -20.0
    assert result["average_robot_utilization"]["percent_change"] == pytest.approx(20.0)
