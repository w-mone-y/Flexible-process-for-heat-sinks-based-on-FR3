"""V2 disturbance flexibility: fault injection, recovery and isolation.

The properties that matter:

*   **Deferred arming** — an operator injects before the relevant operation runs
    and the fault fires when that operation actually starts.  This is V1's best
    idea and the reason the operator never has to time an injection.
*   **Recovery actually costs time** — a rework that does not lengthen makespan
    is indistinguishable from no fault at all, which would make the whole
    disturbance story unfalsifiable.
*   **The line still finishes** — every fault must be survivable; a fault that
    deadlocks the runtime is a bug, not a demonstration.
*   **Retry ceiling** — repeated failures on one target escalate to a human
    instead of looping forever.
"""

from __future__ import annotations

import pytest

from brazing_sim.api import validate_http_command
from brazing_sim.dual_line.application import V2ControlSurface
from brazing_sim.dual_line.faults import FAULT_OPERATION_KINDS, V2FaultController
from brazing_sim.dual_line.presentation import V2StatePresenter
from brazing_sim.dual_line.runtime import DualLineRuntime, UnitStage
from brazing_sim.fault_catalog import MANUAL_FAULT_CATALOG
from brazing_sim.recovery.fault_models import FaultType, RecoveryStatus

# Every fault the V2 console can raise, with a representative target.
FAULT_CASES = (
    ("BRAZING_MISSING", "slot_02_left", {}),
    ("BRAZING_PATH_DEVIATION", "path_03", {}),
    ("FIN_PICK_FAILED", "fin_01", {}),
    ("FIN_GEOMETRY_FAILED", "fin_03", {}),
    ("ARM_UNAVAILABLE", "ARM2", {"duration_s": 5.0}),
    ("ELEVATOR_TIMEOUT", "", {"duration_s": 4.0}),
    ("FORK_TIMEOUT", "", {"duration_s": 4.0}),
    ("FURNACE_DOOR_INTERLOCK", "", {"duration_s": 4.0}),
    ("RACK_LAYER_UNAVAILABLE", "1", {}),
    ("CONTACT_SAFETY_STOP", "", {"duration_s": 3.0}),
    ("TRAY_STATE_INCONSISTENT", "", {"duration_s": 3.0}),
)


def _run(fault: str | None = None, target: str = "", *, limit: int = 30000, **kwargs):
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="FAULT_A", quantity=1)
    if fault is not None:
        runtime.inject_fault(fault, target=target, **kwargs)
    snapshot = runtime.snapshot()
    for _ in range(limit):
        snapshot = runtime.tick(0.05)
        if runtime.complete:
            break
    return runtime, snapshot


@pytest.fixture(scope="module")
def baseline():
    runtime, _ = _run()
    assert runtime.complete
    return runtime.sim_time


# ------------------------------------------------------------------ catalogue


def test_every_runtime_fault_code_has_a_v2_trigger_rule():
    """A catalogue entry with no trigger rule would silently never fire."""

    runtime_faults = {
        definition.runtime_fault for definition in MANUAL_FAULT_CATALOG.values() if definition.runtime_fault
    }
    covered = {item.value for item in FAULT_OPERATION_KINDS}
    assert runtime_faults - covered == set()


def test_fault_catalog_is_exactly_thirteen_explicit_dispositions() -> None:
    """The UI must explain whether each advertised fault is autonomous or human-owned."""

    assert len(MANUAL_FAULT_CATALOG) == 13
    rows = [definition.as_dict() for definition in MANUAL_FAULT_CATALOG.values()]
    assert {row["recovery_class"] for row in rows} == {
        "AUTONOMOUS_RECOVERY",
        "MANUAL_DISPOSITION",
    }
    assert all(row["detection_stage"] for row in rows)
    assert all(row["recovery_route_zh"] for row in rows)
    assert all(row["final_disposition_zh"] for row in rows)
    assert sum(row["recovery_class"] == "AUTONOMOUS_RECOVERY" for row in rows) == 7
    assert sum(row["recovery_class"] == "MANUAL_DISPOSITION" for row in rows) == 6


def test_recovery_snapshot_exposes_policy_and_final_disposition() -> None:
    automatic = V2FaultController()
    auto_record = automatic.inject(
        FaultType.BRAZING_MISSING,
        source="ARM3",
        target="path_02",
        unit_id="U1",
        now=0.0,
        visual_type="BRAZING_MISSING",
    )
    auto_plan = automatic.plans[auto_record.recovery_id]
    assert auto_plan.recovery_class == "AUTONOMOUS_RECOVERY"
    assert auto_plan.final_disposition_zh == "修复后复检，合格则回归原订单"

    manual = V2FaultController()
    manual_record = manual.inject(
        FaultType.ARM_UNAVAILABLE,
        source="ARM2",
        target="ARM2",
        now=0.0,
    )
    manual_plan = manual.plans[manual_record.recovery_id]
    assert manual_plan.recovery_class == "MANUAL_DISPOSITION"
    assert manual_plan.manual_review_complete_at == pytest.approx(10.0)
    assert manual.snapshot()["fault_policy_summary"] == {
        "catalog_count": 13,
        "autonomous_count": 7,
        "manual_count": 6,
        "active_autonomous": 0,
        "active_manual": 1,
        "unresolved_count": 1,
    }


def test_path_targets_accept_both_v1_and_v2_naming():
    """V1 names paths ``slot_02_left``; V2 names them ``path_02``."""

    for target in ("slot_02_left", "path_02"):
        command = validate_http_command("/faults/inject", {"fault_type": "BRAZING_MISSING", "target": target})
        assert command["target"] == target
    with pytest.raises(ValueError):
        validate_http_command("/faults/inject", {"fault_type": "BRAZING_MISSING", "target": "bogus_02"})


def test_removed_fin_insert_fault_cannot_be_armed_directly() -> None:
    runtime = DualLineRuntime(fast=True)
    with pytest.raises(ValueError, match="FIN_INSERT_FAILED"):
        runtime.inject_fault("FIN_INSERT_FAILED", target="fin_02")


# --------------------------------------------------------------- arming model


def test_process_faults_wait_for_their_operation():
    """The operator must not have to time the injection."""

    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="ARM_WAIT", quantity=1)
    request = runtime.inject_fault("BRAZING_MISSING", target="slot_02_left")
    assert request["status"] == "ARMED"
    assert not runtime.faults.faults  # nothing has happened yet
    for _ in range(400):
        runtime.tick(0.05)
        if runtime.faults.faults:
            break
    assert runtime.faults.faults, "故障应在对应工序运行时触发"
    fired = next(iter(runtime.faults.pending.values()))
    assert fired.status == "FIRED"
    assert fired.fired_at is not None and fired.fired_at > fired.armed_at


def test_equipment_faults_fire_immediately():
    """An arm going offline is not tied to a process step."""

    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="IMMEDIATE", quantity=1)
    request = runtime.inject_fault("ARM_UNAVAILABLE", target="ARM2", auto_recover=False)
    assert request["status"] == "FIRED"
    assert not runtime.faults.resource_available("ARM2")


@pytest.mark.parametrize("resource", ("ARM1", "ARM2", "ARM3"))
def test_arm_offline_uses_a_ten_second_simulated_manual_review(resource: str) -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.inject_fault("ARM_UNAVAILABLE", target=resource)
    plan = next(iter(runtime.faults.plans.values()))
    assert plan.status is RecoveryStatus.MANUAL_REVIEW
    notice = runtime.snapshot()["manual_review_notices"][0]
    assert notice["message"] == "发生机械臂暂时离线故障❌，需进行人工审核🔩🔧，请稍作等待⏰"

    runtime.tick(9.99)
    assert not runtime.faults.resource_available(resource)
    assert plan.status is RecoveryStatus.MANUAL_REVIEW

    runtime.tick(0.02)
    assert runtime.faults.resource_available(resource)
    assert plan.status is RecoveryStatus.SUCCEEDED
    assert runtime.snapshot()["manual_review_notices"][0]["message"] == "修改成功✅"


@pytest.mark.parametrize(
    "fault,isolated_resource",
    (("CONTACT_SAFETY_STOP", None), ("TRAY_STATE_INCONSISTENT", "OUTPUT")),
)
def test_safety_fault_manual_review_finishes_after_ten_seconds(
    fault: str,
    isolated_resource: str | None,
) -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id=f"TIMED_{fault}", quantity=1)
    runtime.inject_fault(fault, target="")
    plan = next(iter(runtime.faults.plans.values()))
    assert plan.status is RecoveryStatus.MANUAL_REVIEW
    assert plan.manual_review_complete_at == pytest.approx(plan.created_at + 10.0)

    runtime.tick(9.99)
    assert plan.status is RecoveryStatus.MANUAL_REVIEW
    assert not runtime.faults.cell_available() or isolated_resource in runtime.faults.isolated_resources

    runtime.tick(0.02)
    assert plan.status is RecoveryStatus.SUCCEEDED
    assert plan.message == "修改成功✅"
    assert runtime.faults.cell_available()
    if isolated_resource is not None:
        assert isolated_resource not in runtime.faults.isolated_resources


@pytest.mark.parametrize("fault", ("FIN_PICK_FAILED", "FIN_GEOMETRY_FAILED"))
def test_fifth_and_sixth_faults_enter_manual_review_after_camera_detection(fault: str) -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id=f"MANUAL_{fault}")
    runtime.inject_fault(fault, target="fin_02")
    for _ in range(2_000):
        runtime.tick(0.02)
        if runtime.faults.plans:
            break
    plan = next(iter(runtime.faults.plans.values()))
    assert plan.status is RecoveryStatus.MANUAL_REVIEW
    assert plan.manual_review_complete_at == pytest.approx(plan.created_at + 10.0)


def test_fin_pose_alias_keeps_its_existing_automatic_reseat() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="POSE_STAYS_AUTOMATIC")
    runtime.inject_fault("FIN_POSE", target="fin_02")
    for _ in range(2_000):
        runtime.tick(0.02)
        if runtime.faults.plans:
            break
    plan = next(iter(runtime.faults.plans.values()))
    assert plan.status is RecoveryStatus.RUNNING
    assert plan.strategy == "FIN_REINSTALL"


def test_armed_requests_that_never_match_are_marked_missed():
    controller = V2FaultController()
    controller.arm(FaultType.BRAZING_MISSING, target="slot_09_left", now=0.0)
    controller.expire_armed()
    assert all(item.status == "MISSED" for item in controller.pending.values())


# ------------------------------------------------------------- survivability


@pytest.mark.parametrize("fault,target,kwargs", FAULT_CASES)
def test_every_fault_is_survivable(fault, target, kwargs, baseline):
    """A fault must recover autonomously or through the simulated review."""

    runtime, snapshot = _run(fault, target, **kwargs)
    assert runtime.complete, f"{fault} 导致产线卡死"
    assert snapshot["faults_v2"], f"{fault} 未产生故障记录"
    assert runtime.sim_time >= baseline


@pytest.mark.parametrize(
    "fault,target,kwargs",
    [case for case in FAULT_CASES if case[0] not in {"RACK_LAYER_UNAVAILABLE"}],
)
def test_faults_are_planned_with_a_named_strategy(fault, target, kwargs):
    runtime, snapshot = _run(fault, target, **kwargs)
    recoveries = snapshot["recoveries"]
    assert recoveries, f"{fault} 未生成恢复计划"
    assert recoveries[0]["strategy"]
    assert recoveries[0]["label_zh"], "恢复策略必须有中文说明供 UI 展示"


# --------------------------------------------------------- recovery costs time


@pytest.mark.parametrize(
    "fault,target,kwargs",
    (
        ("BRAZING_MISSING", "slot_02_left", {}),
        ("FIN_GEOMETRY_FAILED", "fin_02", {}),
        ("ARM_UNAVAILABLE", "ARM2", {"duration_s": 5.0}),
        ("ELEVATOR_TIMEOUT", "", {"duration_s": 4.0}),
        ("FURNACE_DOOR_INTERLOCK", "", {"duration_s": 4.0}),
    ),
)
def test_recovery_lengthens_makespan(fault, target, kwargs, baseline):
    """Free recovery would make the disturbance story unfalsifiable."""

    runtime, _ = _run(fault, target, **kwargs)
    assert runtime.sim_time > baseline, f"{fault} 的恢复没有付出任何时间代价"


def test_quality_rework_rewinds_the_stage_machine():
    runtime, snapshot = _run("BRAZING_MISSING", "slot_02_left")
    rollbacks = [event for event in runtime.events if event["type"] == "RECOVERY_ROLLBACK"]
    assert len(rollbacks) == 1
    assert rollbacks[0]["stage"] == UnitStage.MATERIAL_INSPECTION.value
    # Local touch-up must add real recovery time without repeating the entire
    # multi-pass board operation.
    assert 0.0 < rollbacks[0]["effort_factor"] < 1.0
    rework = next(event for event in runtime.events if event["type"] == "REWORK_EFFORT_APPLIED")
    assert rework["strategy"] == "LOCAL_BRAZING_REWORK"
    assert rework["target_index"] == 3  # slot_02_left -> third physical path
    cancelled = [event for event in runtime.events if event["type"] == "OPERATION_CANCELLED"]
    assert cancelled, "回滚前必须取消进行中的操作"
    assert snapshot["recoveries"][0]["status"] == RecoveryStatus.SUCCEEDED.value


def test_fin_pose_rework_redoes_only_the_failed_fin():
    """Rolling back must not discard the fins already installed."""

    runtime, _ = _run("FIN_POSE", "fin_02")
    rollback = next(e for e in runtime.events if e["type"] == "RECOVERY_ROLLBACK")
    installs = [
        event
        for event in runtime.events
        if event["type"] == "FIN_INSTALLED" and event["time"] <= rollback["time"]
    ]
    # One fin is un-done, not the whole set.
    assert rollback["fins_installed"] == max(0, len(installs) - 1)


def test_rework_only_counts_once_the_rollback_ran():
    """A plan that rewound nothing has not recovered from anything."""

    controller = V2FaultController()
    controller.inject(
        FaultType.BRAZING_MISSING,
        source="ARM3",
        target="slot_01_left",
        unit_id="U1",
        now=0.0,
    )
    plan = next(iter(controller.plans.values()))
    assert plan.rollback_stage and not plan.rollback_applied
    controller.complete_recovery("U1", now=1.0)
    assert plan.status is RecoveryStatus.RUNNING, "未执行回滚不得判定为已恢复"
    plan.rollback_applied = True
    controller.complete_recovery("U1", now=2.0)
    assert plan.status is RecoveryStatus.SUCCEEDED


# ------------------------------------------------------------------ isolation


def test_isolated_resource_cannot_start_work():
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="ISO", quantity=1)
    runtime.inject_fault("ARM_UNAVAILABLE", target="ARM2", auto_recover=False)
    for _ in range(180):
        runtime.tick(0.05)
    assert "ARM2" not in runtime.operations, "被隔离的资源不得接受新任务"
    dispensing = [
        event
        for event in runtime.events
        if event["type"] == "OPERATION_STARTED" and event.get("kind") == "DISPENSING"
    ]
    assert not dispensing, "Arm2 离线期间不应开始涂覆"
    for _ in range(40):
        runtime.tick(0.05)
    assert any(
        event["type"] == "OPERATION_STARTED" and event.get("kind") == "DISPENSING" for event in runtime.events
    ), "10秒人工审核结束后Arm2应继续原订单"


def test_arm_manual_review_releases_the_resource_after_ten_seconds():
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="AUTO", quantity=1)
    runtime.inject_fault("ARM_UNAVAILABLE", target="ARM2", duration_s=3.0)
    assert not runtime.faults.resource_available("ARM2")
    while runtime.sim_time < 9.9:
        runtime.tick(0.05)
    assert not runtime.faults.resource_available("ARM2")
    while runtime.sim_time < 10.1:
        runtime.tick(0.05)
    assert runtime.faults.resource_available("ARM2")
    assert any(event["type"] == "RESOURCE_RECOVERED" for event in runtime.events)


def test_active_arm_operation_is_frozen_while_that_arm_is_offline():
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="ACTIVE_ARM_HOLD", quantity=1)
    for _ in range(1_000):
        runtime.tick(0.02)
        operation = runtime.operations.get("ARM2")
        if operation is not None and operation.kind == "DISPENSING":
            break
    else:
        raise AssertionError("Arm2 dispensing did not start")

    runtime.inject_fault("ARM_UNAVAILABLE", target="ARM2", duration_s=2.0)
    before = runtime.operations["ARM2"].remaining_s
    runtime.tick(0.50)
    assert runtime.operations["ARM2"].remaining_s == pytest.approx(before)


def test_manual_arm_review_cannot_be_bypassed_before_its_deadline():
    runtime = DualLineRuntime(fast=True)
    assert runtime.recover_resource("ARM2") is False
    runtime.inject_fault("ARM_UNAVAILABLE", target="ARM2", auto_recover=False)
    assert runtime.recover_resource("ARM2") is False
    runtime.tick(10.0)
    assert runtime.faults.resource_available("ARM2")


def test_rack_layer_isolation_is_reported():
    runtime, snapshot = _run("RACK_LAYER_UNAVAILABLE", "1")
    assert 1 in runtime.faults.unavailable_rack_layers or snapshot["unavailable_rack_layers"]


def test_rack_layer_reallocation_plan_completes_after_another_layer_is_selected():
    runtime, snapshot = _run("RACK_LAYER_UNAVAILABLE", "1")
    assignments = [unit.furnace_layer for unit in runtime.units.values() if unit.furnace_layer is not None]
    assert assignments and 1 not in assignments
    assert snapshot["recoveries"][0]["status"] == RecoveryStatus.SUCCEEDED.value


@pytest.mark.parametrize(
    "fault",
    ("ELEVATOR_TIMEOUT", "FORK_TIMEOUT", "FURNACE_DOOR_INTERLOCK"),
)
def test_mechanism_timeout_holds_the_operation_where_it_occurs_and_then_recovers(fault):
    runtime, snapshot = _run(fault, "", duration_s=2.0)
    injected = next(event for event in runtime.events if event["type"] == "FAULT_INJECTED")
    hold = next(event for event in runtime.events if event["type"] == "FAULT_HOLD_APPLIED")
    assert hold["unit_id"] == injected["unit_id"]
    assert hold["kind"] in {
        "FURNACE_LOAD_TRAY",
        "FURNACE_UNLOAD_TRAY",
        "FURNACE_FRONT_OPEN",
        "FURNACE_FRONT_CLOSE",
        "FURNACE_REAR_OPEN",
        "FURNACE_REAR_CLOSE",
    }
    assert hold["applied_to"] == "CURRENT_OPERATION"
    assert snapshot["recoveries"][0]["status"] == RecoveryStatus.SUCCEEDED.value


@pytest.mark.parametrize(
    "fault,isolated_resource",
    (("CONTACT_SAFETY_STOP", None), ("TRAY_STATE_INCONSISTENT", "OUTPUT")),
)
def test_timed_safety_review_cannot_be_bypassed_by_retry(fault, isolated_resource):
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id=f"SAFE_{fault}", quantity=1)
    runtime.inject_fault(fault, target="", auto_recover=False)
    plan = next(iter(runtime.faults.plans.values()))
    assert plan.status is RecoveryStatus.MANUAL_REVIEW
    assert not runtime.faults.cell_available() or isolated_resource in runtime.faults.isolated_resources

    assert not runtime.recovery_action(plan.recovery_id, "retry")
    runtime.tick(10.01)
    assert plan.status is RecoveryStatus.SUCCEEDED
    assert runtime.faults.cell_available()
    if isolated_resource is not None:
        assert isolated_resource not in runtime.faults.isolated_resources


def test_furnace_profile_is_released_after_the_simulated_manual_review():
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="PROFILE_QUARANTINE", quantity=1)
    runtime.inject_fault("FURNACE_PROFILE", target="furnace", severity="severe")
    for _ in range(10_000):
        runtime.tick(0.05)
        if runtime.faults.faults:
            break
    plan = next(iter(runtime.faults.plans.values()))
    unit = runtime.units["PROFILE_QUARANTINE_UNIT_01"]
    assert plan.status is RecoveryStatus.MANUAL_REVIEW
    for _ in range(190):
        runtime.tick(0.05)
    assert unit.stage is UnitStage.POST_BRAZE_INSPECTION
    assert not any(event["type"] == "UNIT_COMPLETED" for event in runtime.events)
    for _ in range(30_000):
        runtime.tick(0.05)
        if runtime.complete:
            break
    assert plan.status is RecoveryStatus.SUCCEEDED
    assert plan.message == "修改成功✅"
    assert runtime.complete


def test_quarantined_quality_fault_cannot_be_retried_as_fake_automatic_recovery():
    controller = V2FaultController()
    record = controller.inject(
        FaultType.FURNACE_PROFILE,
        source="POST_CAMERA",
        target="furnace",
        unit_id="U1",
        now=0.0,
        severity="severe",
    )
    plan = controller.plans[record.recovery_id]
    assert plan.status is RecoveryStatus.MANUAL_REVIEW
    assert controller.action(plan.recovery_id, "retry", 1.0) is False
    assert plan.status is RecoveryStatus.MANUAL_REVIEW


# ------------------------------------------------------------- retry ceiling


def test_repeated_failures_escalate_to_manual_review():
    """Two attempts per (strategy, unit, target), then a human."""

    controller = V2FaultController()
    statuses = []
    for index in range(4):
        controller.inject(
            FaultType.BRAZING_MISSING,
            source="ARM3",
            target="slot_02_left",
            unit_id="U1",
            now=float(index),
        )
        statuses.append(list(controller.plans.values())[-1].strategy)
    assert statuses[:2] == ["LOCAL_BRAZING_REWORK", "LOCAL_BRAZING_REWORK"]
    assert statuses[2:] == ["MANUAL_REVIEW", "MANUAL_REVIEW"]
    manual_plans = [plan for plan in controller.plans.values() if plan.status is RecoveryStatus.MANUAL_REVIEW]
    assert all(
        plan.manual_review_complete_at == pytest.approx(plan.created_at + 10.0) for plan in manual_plans
    )


def test_operator_manual_review_action_also_uses_the_ten_second_simulation() -> None:
    controller = V2FaultController()
    record = controller.inject(
        FaultType.BRAZING_MISSING,
        source="ARM3",
        target="slot_02_left",
        unit_id="U1",
        now=0.0,
    )
    plan = controller.plans[record.recovery_id]
    assert controller.action(plan.recovery_id, "manual_review", 2.0)
    assert plan.status is RecoveryStatus.MANUAL_REVIEW
    assert plan.manual_review_complete_at == pytest.approx(12.0)
    assert not controller.service_manual_reviews(11.99)
    assert controller.service_manual_reviews(12.01) == [plan]
    assert plan.message == "修改成功✅"


@pytest.mark.parametrize(
    "fault",
    (FaultType.ELEVATOR_TIMEOUT, FaultType.FORK_TIMEOUT, FaultType.FURNACE_DOOR_INTERLOCK),
)
def test_repeated_mechanism_timeout_escalates_after_its_single_safe_retry(fault):
    controller = V2FaultController()
    first = controller.inject(
        fault,
        source="FURNACE_TRANSFER",
        unit_id="U1",
        now=0.0,
    )
    assert controller.plans[first.recovery_id].status is RecoveryStatus.RUNNING

    second = controller.inject(
        fault,
        source="FURNACE_TRANSFER",
        unit_id="U1",
        now=1.0,
    )
    assert controller.plans[second.recovery_id].status is RecoveryStatus.MANUAL_REVIEW


def test_repeated_runtime_mechanism_failure_uses_timed_manual_review_before_resuming():
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="REPEATED_TRANSFER", quantity=1)
    runtime.inject_fault("ELEVATOR_TIMEOUT", target="", duration_s=1.0)
    runtime.inject_fault("ELEVATOR_TIMEOUT", target="", duration_s=1.0)

    held_operation = None
    manual_plan = None
    for _ in range(10_000):
        runtime.tick(0.05)
        manual_plan = next(
            (plan for plan in runtime.faults.plans.values() if plan.status is RecoveryStatus.MANUAL_REVIEW),
            None,
        )
        held_operation = next(
            (operation for operation in runtime.operations.values() if operation.manual_hold_fault_ids),
            None,
        )
        if manual_plan is not None and held_operation is not None:
            break
    assert manual_plan is not None and held_operation is not None

    before = held_operation.remaining_s
    runtime.tick(1.0)
    assert held_operation.remaining_s == pytest.approx(before)
    assert not runtime.recovery_action(manual_plan.recovery_id, "retry")
    runtime.tick(9.05)
    assert manual_plan.status is RecoveryStatus.SUCCEEDED

    for _ in range(30_000):
        runtime.tick(0.05)
        if runtime.complete:
            break
    assert runtime.complete


def test_safety_faults_go_straight_to_manual_review():
    for fault in ("CONTACT_SAFETY_STOP", "TRAY_STATE_INCONSISTENT"):
        runtime = DualLineRuntime(fast=True)
        runtime.inject_fault(fault, target="", duration_s=3.0)
        plan = next(iter(runtime.faults.plans.values()))
        assert plan.strategy == "MANUAL_REVIEW"
        assert plan.status is RecoveryStatus.MANUAL_REVIEW
        assert plan.manual_review_complete_at == pytest.approx(10.0)


def test_recovery_actions_apply_and_respect_the_retry_limit():
    controller = V2FaultController()
    controller.inject(FaultType.BRAZING_MISSING, source="ARM3", target="p", unit_id="U1", now=0.0)
    plan = next(iter(controller.plans.values()))
    assert controller.action(plan.recovery_id, "pause", 1.0)
    assert plan.status is RecoveryStatus.PAUSED
    assert controller.action(plan.recovery_id, "resume", 2.0)
    assert plan.status is RecoveryStatus.RUNNING
    # ``retry_limit`` retries are allowed; the next one escalates.
    for attempt in range(plan.retry_limit):
        assert controller.action(plan.recovery_id, "retry", 3.0 + attempt)
    assert plan.retry_count == plan.retry_limit
    assert controller.action(plan.recovery_id, "retry", 9.0) is False
    assert plan.status is RecoveryStatus.MANUAL_REVIEW
    assert controller.action("NOPE", "pause", 10.0) is False


# ----------------------------------------------------------- console wiring


def test_console_commands_reach_the_runtime():
    """The full path: HTTP validation → command → runtime → snapshot."""

    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="WIRE", quantity=1)
    surface = V2ControlSurface(runtime)
    surface.process(
        validate_http_command(
            "/faults/inject",
            {"fault_type": "ARM_UNAVAILABLE", "target": "ARM2", "auto_recover": False},
        )
    )
    runtime.tick(0.05)
    state = V2StatePresenter().present(runtime.snapshot(), simulation_speed=1.0, actual_rtf=1.0)
    assert state["ui_capabilities"]["fault_injection"] is True
    assert state["faults_v2"] and state["recoveries"]
    assert state["resources_v2"]["ARM2"]["status"] == "FAULTED"
    assert state["resources_v2"]["ARM2"]["fault_code"] == "ARM_UNAVAILABLE"

    recovery_id = state["recoveries"][0]["recovery_id"]
    surface.process({"type": "recovery_action", "recovery_id": recovery_id, "action": "manual_review"})
    with pytest.raises(ValueError):
        surface.process({"type": "resource_recover", "resource_id": "ARM2"})
    runtime.tick(10.0)
    state = V2StatePresenter().present(runtime.snapshot(), simulation_speed=1.0, actual_rtf=1.0)
    assert state["resources_v2"]["ARM2"]["status"] != "FAULTED"


def test_unsupported_commands_are_still_rejected():
    surface = V2ControlSurface(DualLineRuntime(fast=True))
    with pytest.raises(ValueError):
        surface.process({"type": "scheduler_replan"})
    with pytest.raises(ValueError):
        surface.process({"type": "resource_recover", "resource_id": "ARM1"})


def test_snapshot_reports_recovery_rate():
    _, snapshot = _run("BRAZING_MISSING", "slot_02_left")
    metrics = snapshot["metrics"]
    assert metrics["fault_count"] == 1
    assert metrics["recovered_fault_count"] == 1
    assert metrics["recovery_rate"] == pytest.approx(1.0)


def test_reset_clears_all_fault_state():
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="RST", quantity=1)
    runtime.inject_fault("ARM_UNAVAILABLE", target="ARM2", auto_recover=False)
    runtime.reset()
    assert not runtime.faults.faults
    assert not runtime.faults.pending
    assert runtime.faults.resource_available("ARM2")
    assert runtime.snapshot()["faults_v2"] == []
