from __future__ import annotations

import pytest

from brazing_sim.experiments import MetricsCollector, compare_experiments
from brazing_sim.flexible import build_inline_plan, build_preset_plan
from brazing_sim.dual_line.runtime import DualLineRuntime
from brazing_sim.dual_line.scene_adapter import DualLineSceneAdapter
from brazing_sim.dual_line.unified_runtime import UnifiedV2Runtime
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


@pytest.mark.parametrize("fault_type", ("BRAZING_MISSING", "BRAZING_PATH_DEVIATION"))
def test_multi_order_brazing_rework_waits_for_s2a_instead_of_bypassing_return(
    fault_type: str,
) -> None:
    """A busy S2A must delay the faulty tray, never send it forward to S3."""

    runtime = DualLineRuntime(fast=True)
    for preset in ("A", "B", "C"):
        runtime.submit_order(preset, order_id=f"MULTI_{preset}")
    runtime.inject_fault(fault_type, target="path_02")

    for _ in range(20_000):
        runtime.tick(0.02)
        if runtime.complete:
            break
    assert runtime.complete

    faulty_unit = runtime.units["MULTI_A_UNIT_01"]
    recovery_return = next(
        event
        for event in runtime.events
        if event["type"] == "RECOVERY_RETURN_STARTED" and event["unit_id"] == faulty_unit.unit_id
    )
    assert recovery_return["source"] == "S2B"
    assert recovery_return["target"] == "S2A"

    forward_handoffs = [
        event
        for event in runtime.events
        if event["type"] == "TRAY_HANDOFF"
        and event["unit_id"] == faulty_unit.unit_id
        and event["source"] == "S2B"
        and event["target"] in {"INSTALL_A", "INSTALL_B"}
    ]
    assert forward_handoffs
    assert all(event["time"] > recovery_return["time"] for event in forward_handoffs)

    # The unrelated orders are allowed to progress while A waits for S2A.
    assert any(
        event["type"] == "OPERATION_COMPLETED"
        and event["unit_id"] != faulty_unit.unit_id
        and event["time"] < recovery_return["time"]
        for event in runtime.events
    )


def _s1_to_s2a_handoff_time(runtime: DualLineRuntime, unit_id: str) -> float:
    return next(
        float(event["time"])
        for event in runtime.events
        if event["type"] == "TRAY_HANDOFF"
        and event["unit_id"] == unit_id
        and event["source"] == "S1"
        and event["target"] == "S2A"
    )


def _material_inspection_pass_time(runtime: DualLineRuntime, unit_id: str) -> float:
    return max(
        float(event["time"])
        for event in runtime.events
        if event["type"] == "OPERATION_COMPLETED"
        and event["unit_id"] == unit_id
        and event["kind"] == "MATERIAL_INSPECTION"
    )


def test_next_board_enters_s2a_only_after_previous_brazing_inspection_passes() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="INSPECTION_GATE_A")
    runtime.submit_order("B", order_id="INSPECTION_GATE_B")

    for _ in range(4_000):
        runtime.tick(0.02)
        if runtime.complete:
            break

    assert runtime.complete
    inspection_passed_at = _material_inspection_pass_time(runtime, "INSPECTION_GATE_A_UNIT_01")
    next_board_released_at = _s1_to_s2a_handoff_time(runtime, "INSPECTION_GATE_B_UNIT_01")
    assert inspection_passed_at <= next_board_released_at
    assert next_board_released_at - inspection_passed_at <= 0.02


def test_next_board_waits_for_brazing_rework_reinspection_before_entering_s2a() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="REWORK_GATE_A")
    runtime.submit_order("B", order_id="REWORK_GATE_B")
    runtime.inject_fault("BRAZING_MISSING", target="path_02")

    for _ in range(4_000):
        runtime.tick(0.02)
        if runtime.complete:
            break

    assert runtime.complete
    assert any(
        event["type"] == "RECOVERY_RETURN_STARTED" and event.get("unit_id") == "REWORK_GATE_A_UNIT_01"
        for event in runtime.events
    )
    reinspection_passed_at = _material_inspection_pass_time(runtime, "REWORK_GATE_A_UNIT_01")
    next_board_released_at = _s1_to_s2a_handoff_time(runtime, "REWORK_GATE_B_UNIT_01")
    assert reinspection_passed_at <= next_board_released_at
    assert next_board_released_at - reinspection_passed_at <= 0.02


def test_unified_mujoco_brazing_rework_preserves_bridge_and_releases_next_board() -> None:
    pytest.importorskip("mujoco")
    runtime = UnifiedV2Runtime(fast=True)
    scene = DualLineSceneAdapter("scenes/production/brazing_line_v2.xml")
    runtime.set_execution_gate(scene)
    runtime.submit_order("A", order_id="MUJOCO_REWORK_A")
    runtime.submit_order("B", order_id="MUJOCO_REWORK_B")
    runtime.inject_fault("BRAZING_MISSING", target="path_02")

    try:
        for _ in range(4_000):
            runtime.tick(0.05)
            scene.sync(runtime.physical_runtime)
            scene.step_physics(0.05)
            if runtime.complete and scene.transport_settled:
                break

        assert runtime.complete
        assert scene.transport_settled
        assert runtime.physical_runtime._execution_gate is runtime.bridge
        assert any(
            event["type"] == "RECOVERY_RETURN_STARTED" and event.get("unit_id") == "MUJOCO_REWORK_A_UNIT_01"
            for event in runtime.physical_runtime.events
        )
        reinspection_passed_at = _material_inspection_pass_time(
            runtime.physical_runtime,
            "MUJOCO_REWORK_A_UNIT_01",
        )
        next_board_released_at = _s1_to_s2a_handoff_time(
            runtime.physical_runtime,
            "MUJOCO_REWORK_B_UNIT_01",
        )
        assert reinspection_passed_at <= next_board_released_at
        assert next_board_released_at - reinspection_passed_at <= 0.05
    finally:
        scene.close()


def test_unified_mujoco_fin_pick_failure_returns_to_arm3_and_reinspects() -> None:
    """The production runtime must authorize Arm3's physical one-fin recovery."""

    pytest.importorskip("mujoco")
    runtime = UnifiedV2Runtime(fast=True)
    scene = DualLineSceneAdapter("scenes/production/brazing_line_v2.xml")
    runtime.set_execution_gate(scene)
    runtime.submit_order("A", order_id="MUJOCO_FIN_REWORK_A")
    runtime.submit_order("B", order_id="MUJOCO_FIN_REWORK_B")
    runtime.submit_order("C", order_id="MUJOCO_FIN_REWORK_C")
    runtime.inject_fault("FIN_PICK_FAILED", target="fin_02")

    try:
        for _ in range(8_000):
            runtime.tick(0.05)
            scene.sync(runtime.physical_runtime)
            scene.step_physics(0.05)
            if runtime.complete and scene.transport_settled:
                break

        assert runtime.complete
        assert scene.transport_settled
        assert runtime.physical_runtime._execution_gate is runtime.bridge
        assert not runtime.physical_runtime.snapshot()["manual_review_notices"]
        faulty_unit_id = "MUJOCO_FIN_REWORK_A_UNIT_01"
        return_event = next(
            event
            for event in runtime.physical_runtime.events
            if event["type"] == "RECOVERY_RETURN_STARTED"
            and event.get("unit_id") == faulty_unit_id
        )
        assert return_event["source"] == "S4"
        assert return_event["target"] == "INSTALL_B"
        assert not any(
            event["type"] == "TRAY_HANDOFF"
            and event.get("unit_id") != faulty_unit_id
            and event.get("target") == "INSTALL_B"
            and event["time"] < return_event["time"]
            for event in runtime.physical_runtime.events
        )
        inspections = [
            event
            for event in runtime.physical_runtime.events
            if event["type"] == "OPERATION_STARTED"
            and event.get("unit_id") == faulty_unit_id
            and event.get("kind") == "PRE_BRAZE_INSPECTION"
        ]
        assert len(inspections) == 2
        faulty_unit = runtime.physical_runtime.units[faulty_unit_id]
        assert faulty_unit.fins_installed == faulty_unit.fin_count
    finally:
        scene.close()


def test_fin_pick_recovery_clears_an_occupied_s3b_without_deadlock() -> None:
    """A blocked S4 inspection must not reserve Arm3 forever."""

    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="S3B_DEADLOCK_FAULT")
    runtime.submit_order("B", order_id="S3B_DEADLOCK_MERGE")
    runtime.submit_order("C", order_id="S3B_DEADLOCK_OCCUPANT")
    runtime.inject_fault("FIN_PICK_FAILED", target="fin_02")

    for _ in range(30_000):
        runtime.tick(0.02)
        if runtime.complete:
            break

    assert runtime.complete
    recovery_return = next(
        event
        for event in runtime.events
        if event["type"] == "RECOVERY_RETURN_STARTED"
        and event.get("unit_id") == "S3B_DEADLOCK_FAULT_UNIT_01"
    )
    assert not any(
        event["type"] == "TRAY_HANDOFF"
        and event.get("unit_id") != "S3B_DEADLOCK_FAULT_UNIT_01"
        and event.get("target") == "INSTALL_B"
        and event["time"] < recovery_return["time"]
        for event in runtime.events
    )
    assert recovery_return["source"] == "S4"
    assert recovery_return["target"] == "INSTALL_B"
    assert not runtime.snapshot()["manual_review_notices"]


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
