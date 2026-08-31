from __future__ import annotations

from brazing_sim.flexible import build_inline_plan
from brazing_sim.dual_line import DualLineRuntime, InstallBranch, TrayPhase, UnitStage
from brazing_sim.dual_line.tray_flow import TrayOwner
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.planning import ManufacturingTask, TaskType
from brazing_sim.scheduling.arm1_tool_policy import (
    Arm1OpportunityContext,
    Arm1ToolPolicyConfig,
    Arm1ToolResidencyPolicy,
)
from brazing_sim.scheduling.rolling_horizon import (
    HorizonAction,
    HorizonDecisionContext,
    RollingHorizonPlanner,
)


def test_rolling_horizon_rejects_the_locally_fast_action_that_delays_critical_work() -> None:
    planner = RollingHorizonPlanner(horizon_seconds=45.0, maximum_candidates=6)

    decision = planner.choose(
        (
            HorizonAction(
                action_id="INSTALL_CURRENT_FIN",
                action_zh="立即安装当前翅片",
                duration_s=4.0,
                projected_completion_s=18.0,
                blocks_resource="ARM3",
            ),
            HorizonAction(
                action_id="RESERVE_INSPECTION",
                action_zh="先为到达的检测任务让出Arm3",
                duration_s=6.0,
                projected_completion_s=20.0,
            ),
        ),
        HorizonDecisionContext(
            now=10.0,
            critical_resource="ARM3",
            critical_work_ready_in_s=2.0,
            critical_work_duration_s=5.0,
            critical_path_weight=2.0,
        ),
    )

    assert decision.selected_action_id == "RESERVE_INSPECTION"
    candidates = {item.action_id: item for item in decision.candidates}
    assert candidates["INSTALL_CURRENT_FIN"].critical_path_delay_s == 2.0
    assert candidates["INSTALL_CURRENT_FIN"].total_cost > candidates["RESERVE_INSPECTION"].total_cost
    assert decision.horizon_seconds == 45.0
    assert "关键工序" in decision.explanation_zh


def test_runtime_snapshot_explains_where_tasks_spent_their_waiting_time() -> None:
    runtime = ManufacturingRuntime(
        scheduler_mode="dynamic",
        flexible_cell=True,
        enable_motion_planning=False,
    )
    for index, preset in enumerate(("A", "B", "C"), start=1):
        runtime.submit_plan(
            build_inline_plan(
                preset=preset,
                order_id=f"BOTTLENECK_{index}",
                quantity=1,
                priority=10,
            ),
            now=0.0,
        )

    for tick in range(240):
        runtime.tick((tick + 1) * 0.25)

    report = runtime.snapshot()["bottlenecks"]

    assert report["measurement"] == "task_wait_seconds"
    assert report["total_wait_s"] > 0.0
    assert report["by_reason"]
    assert report["top_tasks"]
    assert all(item["wait_s"] > 0.0 for item in report["by_reason"])
    assert all(
        {"task_id", "display_name_zh", "reason", "wait_s"} <= item.keys() for item in report["top_tasks"]
    )
    assert sum(item["wait_s"] for item in report["by_reason"]) == report["total_wait_s"]


class _ArrivalForecastGate:
    def __init__(self, ready_in_s: float) -> None:
        self.ready_in_s = ready_in_s

    def tray_ready(self, *_args) -> bool:
        return False

    def estimated_tray_ready_in(self, *_args) -> float:
        return self.ready_in_s

    def owner_available(self, *_args) -> bool:
        return True

    def operation_complete(self, *_args) -> bool:
        return False

    def operation_start_allowed(self, *_args) -> bool:
        return True


def test_arm3_reserves_a_future_inspection_window_without_preempting_a_started_fin() -> None:
    runtime = DualLineRuntime(fast=True)
    install_order = runtime.submit_order("A", order_id="ARM3_INSTALL_WINDOW")
    install = runtime.units[install_order.unit_ids[0]]
    assert install.tray_id is not None

    runtime.flow.handoff(
        install.tray_id,
        TrayOwner.S1,
        TrayOwner.S2A,
        TrayPhase.DISPENSING,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        install.tray_id,
        TrayOwner.S2A,
        TrayOwner.S2B,
        TrayPhase.MATERIAL_INSPECTION,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        install.tray_id,
        TrayOwner.S2B,
        TrayOwner.INSTALL_B,
        TrayPhase.FIN_INSTALLATION,
        now=runtime.sim_time,
    )
    install.stage = UnitStage.FIN_INSTALLATION
    install.branch = InstallBranch.ARM3_B

    inspect_order = runtime.submit_order("B", order_id="ARM3_INSPECT_WINDOW")
    inspect = runtime.units[inspect_order.unit_ids[0]]
    assert inspect.tray_id is not None
    runtime.flow.handoff(
        inspect.tray_id,
        TrayOwner.S1,
        TrayOwner.S2A,
        TrayPhase.DISPENSING,
        now=runtime.sim_time,
    )
    inspect.stage = UnitStage.WAITING_S2B
    gate = _ArrivalForecastGate(ready_in_s=0.8)
    runtime.set_execution_gate(gate)
    runtime.operations.clear()

    windows = runtime.snapshot()["arm3_inspection_windows"]
    assert windows == [
        {
            "unit_id": inspect.unit_id,
            "inspection_kind": "MATERIAL_INSPECTION",
            "source_stage": "WAITING_S2B",
            "start_at": 0.8,
            "end_at": 1.115,
            "reason_zh": "托盘预计到达S2B，预留Arm3钎料检测时间窗",
        }
    ]

    # One fast-mode fin needs 0.385 s. It fits before the future window and
    # remains non-preemptible after it starts.
    gate.ready_in_s = 0.8
    gate.tray_ready = lambda tray_id, owner: tray_id == install.tray_id
    runtime.tick(0.05)
    assert runtime.operations["ARM3"].kind == "INSTALL_FIN"
    gate.ready_in_s = 0.1
    runtime.tick(0.05)
    assert runtime.operations["ARM3"].kind == "INSTALL_FIN"


def test_arm3_does_not_start_a_fin_that_would_cross_the_next_inspection_window() -> None:
    runtime = DualLineRuntime(fast=True)
    install_order = runtime.submit_order("A", order_id="ARM3_FIN_BLOCKED")
    install = runtime.units[install_order.unit_ids[0]]
    assert install.tray_id is not None

    runtime.flow.handoff(install.tray_id, TrayOwner.S1, TrayOwner.S2A, TrayPhase.DISPENSING, now=0.0)
    runtime.flow.handoff(
        install.tray_id,
        TrayOwner.S2A,
        TrayOwner.S2B,
        TrayPhase.MATERIAL_INSPECTION,
        now=0.0,
    )
    runtime.flow.handoff(
        install.tray_id,
        TrayOwner.S2B,
        TrayOwner.INSTALL_B,
        TrayPhase.FIN_INSTALLATION,
        now=0.0,
    )
    install.stage = UnitStage.FIN_INSTALLATION
    install.branch = InstallBranch.ARM3_B
    inspect_order = runtime.submit_order("B", order_id="ARM3_WINDOW_BLOCKER")
    inspect = runtime.units[inspect_order.unit_ids[0]]
    assert inspect.tray_id is not None
    runtime.flow.handoff(inspect.tray_id, TrayOwner.S1, TrayOwner.S2A, TrayPhase.DISPENSING, now=0.0)
    inspect.stage = UnitStage.WAITING_S2B
    gate = _ArrivalForecastGate(ready_in_s=0.2)
    gate.tray_ready = lambda tray_id, owner: tray_id == install.tray_id
    runtime.set_execution_gate(gate)
    runtime.operations.clear()

    runtime.tick(0.05)

    assert "ARM3" not in runtime.operations
    metrics = runtime.snapshot()["metrics"]
    assert metrics["arm3_inspection_reservation_wait_s"] > 0.0


def _arm1_task(
    task_id: str,
    task_type: TaskType,
    *,
    ready_at: float,
    duration: float,
) -> ManufacturingTask:
    return ManufacturingTask(
        task_id=task_id,
        task_type=task_type,
        order_id="OPPORTUNITY_ORDER",
        unit_id="OPPORTUNITY_UNIT_01",
        eligible_resources=["ARM1"],
        estimated_duration=duration,
        ready_at=ready_at,
    )


def test_arm1_opportunity_cost_can_switch_before_the_soft_microbatch_limit() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(max_base_microbatch=2, drain_admitted_base_wave=False),
        initial_tool="vacuum_gripper",
    )
    base = _arm1_task("BASE", TaskType.PICK_BASE_PLATE, ready_at=20.0, duration=12.0)
    fin = _arm1_task("FIN", TaskType.PICK_FIN, ready_at=0.0, duration=8.0)

    selection = policy.select(
        (base, fin),
        now=20.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=0.0,
            next_fin_ready_in=0.0,
            base_work_seconds=12.0,
            fin_work_seconds=8.0,
            tool_change_seconds=12.0,
            downstream_blocking_seconds=8.0,
        ),
    )

    assert selection.preferred_tool == "parallel_gripper"
    assert selection.switch_gripper_cost < selection.keep_vacuum_cost
    assert (base.task_id, "ARM1") in selection.blocked_pairs
    snapshot = policy.snapshot()
    assert snapshot["selected_action"] == "SWITCH_GRIPPER"
    assert snapshot["keep_vacuum_cost"] == selection.keep_vacuum_cost
    assert snapshot["switch_gripper_cost"] == selection.switch_gripper_cost
    assert snapshot["rolling_horizon"]["selected_action_id"] == "SWITCH_GRIPPER"
    assert len(snapshot["rolling_horizon"]["candidates"]) == 2


def test_arm1_opportunity_cost_keeps_vacuum_when_no_fin_is_executable() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(max_base_microbatch=2),
        initial_tool="vacuum_gripper",
    )
    policy.base_units_in_residency = 2
    base = _arm1_task("BASE_ONLY", TaskType.PICK_BASE_PLATE, ready_at=0.0, duration=12.0)

    selection = policy.select(
        (base,),
        now=0.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=0.0,
            next_fin_ready_in=float("inf"),
            base_work_seconds=12.0,
            fin_work_seconds=8.0,
            tool_change_seconds=12.0,
        ),
    )

    assert selection.preferred_tool == "vacuum_gripper"
    assert selection.selected_action == "KEEP_VACUUM"
    assert selection.switch_gripper_cost == float("inf")
    assert policy.snapshot()["switch_gripper_cost"] is None


def test_arm1_tool_policy_reset_clears_rolling_horizon_history() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(max_base_microbatch=2),
        initial_tool="vacuum_gripper",
    )
    base = _arm1_task("BASE_RESET", TaskType.PICK_BASE_PLATE, ready_at=0.0, duration=12.0)
    fin = _arm1_task("FIN_RESET", TaskType.PICK_FIN, ready_at=0.0, duration=8.0)
    policy.select(
        (base, fin),
        now=0.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=0.0,
            next_fin_ready_in=0.0,
            base_work_seconds=12.0,
            fin_work_seconds=8.0,
            tool_change_seconds=12.0,
        ),
    )

    policy.reset(initial_tool="vacuum_gripper")

    snapshot = policy.snapshot()
    assert snapshot["selected_action"] == "IDLE"
    assert snapshot["rolling_horizon"]["selected_action_id"] is None
    assert snapshot["rolling_horizon"]["candidates"] == []


def test_arm1_drains_the_admitted_base_wave_before_switching_to_fins() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(
            max_base_microbatch=6,
            drain_admitted_base_wave=True,
        ),
        initial_tool="vacuum_gripper",
    )
    policy.base_units_in_residency = 2
    base = _arm1_task("BASE_WAVE", TaskType.PICK_BASE_PLATE, ready_at=0.0, duration=12.0)
    fin = _arm1_task("FIN_WAITING", TaskType.PICK_FIN, ready_at=0.0, duration=4.0)

    selection = policy.select(
        (base, fin),
        now=35.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=0.0,
            next_fin_ready_in=0.0,
            base_work_seconds=12.0,
            fin_work_seconds=4.0,
            tool_change_seconds=8.0,
            admitted_base_units_remaining=3,
        ),
    )

    assert selection.preferred_tool == "vacuum_gripper"
    assert selection.selected_action == "DRAIN_BASE_WAVE"
    assert (fin.task_id, "ARM1") in selection.blocked_pairs
    assert "3块已接纳主板" in selection.explanation_zh
    snapshot = policy.snapshot()
    assert snapshot["drain_admitted_base_wave"] is True
    assert snapshot["admitted_base_units_remaining"] == 3
    assert snapshot["rolling_horizon"]["selected_action_id"] == "DRAIN_BASE_WAVE"


def test_arm1_base_wave_is_one_buffer_ahead_of_parallel_fin_branches() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(max_base_microbatch=6, drain_admitted_base_wave=True),
        initial_tool="vacuum_gripper",
    )
    base = _arm1_task("BASE_BRANCH_BUFFER", TaskType.PICK_BASE_PLATE, ready_at=0.0, duration=12.0)
    fin = _arm1_task("FIN_BRANCH_BUFFER", TaskType.PICK_FIN, ready_at=0.0, duration=4.0)

    policy.base_units_in_residency = 2
    keep_vacuum = policy.select(
        (base, fin),
        now=10.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=0.0,
            next_fin_ready_in=0.0,
            base_work_seconds=12.0,
            fin_work_seconds=4.0,
            tool_change_seconds=8.0,
            admitted_base_units_remaining=4,
            parallel_fin_branches=2,
        ),
    )
    assert keep_vacuum.selected_action == "DRAIN_BASE_WAVE"

    policy.base_units_in_residency = 3
    release_fins = policy.select(
        (base, fin),
        now=20.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=0.0,
            next_fin_ready_in=0.0,
            base_work_seconds=12.0,
            fin_work_seconds=4.0,
            tool_change_seconds=8.0,
            admitted_base_units_remaining=3,
            parallel_fin_branches=2,
        ),
    )

    assert release_fins.selected_action == "SWITCH_GRIPPER"
    assert release_fins.preferred_tool == "parallel_gripper"
    assert policy.snapshot()["base_wave_target"] == 3


def test_arm3_claiming_its_fin_branch_does_not_restart_arm1_base_wave() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(max_base_microbatch=6, drain_admitted_base_wave=True),
        initial_tool="vacuum_gripper",
    )
    policy.base_units_in_residency = 3
    arm3_fin = _arm1_task("ARM3_BRANCH_FIN", TaskType.PICK_FIN, ready_at=0.0, duration=4.0)

    policy.observe_started(arm3_fin, "ARM3")

    assert policy.base_units_in_residency == 3


def test_arm1_uses_single_base_replenishment_after_the_fin_wave_has_started() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(max_base_microbatch=6, drain_admitted_base_wave=True),
        initial_tool="parallel_gripper",
    )
    first_fin = _arm1_task("FIRST_FIN_WAVE", TaskType.PICK_FIN, ready_at=0.0, duration=4.0)
    policy.observe_started(first_fin, "ARM1")
    policy.active_fin_unit = None
    policy.current_tool = "vacuum_gripper"
    policy.base_units_in_residency = 1
    base = _arm1_task("REPLENISH_BASE", TaskType.PICK_BASE_PLATE, ready_at=0.0, duration=12.0)
    fin = _arm1_task("WAITING_FIN_TRAY", TaskType.PICK_FIN, ready_at=0.0, duration=4.0)

    selection = policy.select(
        (base, fin),
        now=20.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=0.0,
            next_fin_ready_in=0.0,
            base_work_seconds=12.0,
            fin_work_seconds=4.0,
            tool_change_seconds=8.0,
            admitted_base_units_remaining=2,
            parallel_fin_branches=2,
        ),
    )

    assert selection.selected_action == "SWITCH_GRIPPER"
    assert policy.snapshot()["base_wave_target"] == 1


def test_normal_order_priority_cannot_bypass_a_ready_s1_base_wave() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(max_base_microbatch=6, drain_admitted_base_wave=True),
        initial_tool="vacuum_gripper",
    )
    base = _arm1_task("BASE_READY_AT_S1", TaskType.PICK_BASE_PLATE, ready_at=0.0, duration=12.0)
    base.priority = 10
    fin = _arm1_task("LATER_NORMAL_FIN", TaskType.PICK_FIN, ready_at=0.0, duration=4.0)
    fin.priority = 99

    selection = policy.select(
        (base, fin),
        now=5.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=0.0,
            next_fin_ready_in=0.0,
            base_work_seconds=12.0,
            fin_work_seconds=4.0,
            tool_change_seconds=8.0,
            admitted_base_units_remaining=1,
        ),
    )

    assert selection.selected_action == "DRAIN_BASE_WAVE"
    assert selection.preferred_tool == "vacuum_gripper"
    assert (fin.task_id, "ARM1") in selection.blocked_pairs


def test_explicit_urgent_fin_can_use_the_next_safe_wave_boundary() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(max_base_microbatch=4, drain_admitted_base_wave=True),
        initial_tool="vacuum_gripper",
    )
    base = _arm1_task("NORMAL_BASE", TaskType.PICK_BASE_PLATE, ready_at=0.0, duration=12.0)
    fin = _arm1_task("URGENT_FIN", TaskType.PICK_FIN, ready_at=0.0, duration=4.0)
    fin.payload["urgent_order"] = True

    selection = policy.select(
        (base, fin),
        now=5.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=0.0,
            next_fin_ready_in=0.0,
            base_work_seconds=12.0,
            fin_work_seconds=4.0,
            tool_change_seconds=8.0,
            admitted_base_units_remaining=1,
        ),
    )

    assert selection.selected_action == "SWITCH_GRIPPER"
    assert (base.task_id, "ARM1") in selection.blocked_pairs


def test_arm1_does_not_idle_for_a_base_wave_that_cannot_reach_s1() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(
            max_base_microbatch=6,
            drain_admitted_base_wave=True,
        ),
        initial_tool="vacuum_gripper",
    )
    fin = _arm1_task("FIN_CLEAR_BLOCKAGE", TaskType.PICK_FIN, ready_at=0.0, duration=4.0)

    selection = policy.select(
        (fin,),
        now=5.0,
        resource_tool="vacuum_gripper",
        opportunity=Arm1OpportunityContext(
            next_base_ready_in=float("inf"),
            next_fin_ready_in=0.0,
            base_work_seconds=12.0,
            fin_work_seconds=4.0,
            tool_change_seconds=8.0,
            admitted_base_units_remaining=3,
        ),
    )

    assert selection.preferred_tool == "parallel_gripper"
    assert selection.selected_action == "SWITCH_GRIPPER"
