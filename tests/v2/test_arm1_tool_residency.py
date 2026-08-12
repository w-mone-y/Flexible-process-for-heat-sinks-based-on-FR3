from __future__ import annotations

from dataclasses import replace

from brazing_sim.events import EventType
from brazing_sim.dual_line import UnifiedV2Runtime
from brazing_sim.flexible import build_inline_plan
from brazing_sim.manufacturing_config import load_scheduler_config
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.paths import CONFIG_DIR
from brazing_sim.planning import TaskType
from brazing_sim.planning import ManufacturingTask
from brazing_sim.scheduling import Arm1ToolPolicyConfig, Arm1ToolResidencyPolicy


class _InstantPhysicalGate:
    """Measured execution seam without MuJoCo rendering overhead."""

    def tray_ready(self, *_args) -> bool:
        return True

    def owner_available(self, *_args) -> bool:
        return True

    def operation_complete(self, *_args) -> bool:
        return True

    def operation_start_allowed(self, *_args) -> bool:
        return True

    def operation_milestone(self, *_args) -> bool:
        return True


def _run_three_order_mix(
    policy: Arm1ToolPolicyConfig | None = None,
) -> tuple[ManufacturingRuntime, dict[str, object]]:
    scheduler_config = load_scheduler_config(CONFIG_DIR / "scheduler.yaml")
    if policy is not None:
        scheduler_config = replace(scheduler_config, arm1_tool_policy=policy)
    runtime = ManufacturingRuntime(
        scheduler_mode="dynamic",
        flexible_cell=True,
        scheduler_config=scheduler_config,
    )
    for index, preset in enumerate(("A", "B", "C"), start=1):
        runtime.submit_plan(
            build_inline_plan(
                preset=preset,
                order_id=f"ARM1_RESIDENCY_{index}",
                quantity=1,
                priority=10,
            ),
            now=0.0,
        )
    for tick in range(4_000):
        runtime.tick((tick + 1) * 0.25)
        if runtime.terminal:
            break
    else:
        raise AssertionError("three-order tool-residency scenario did not complete")
    return runtime, runtime.snapshot()


def _arm1_events(runtime: ManufacturingRuntime):
    return [
        event
        for event in runtime.events.history
        if event.source == "ARM1" and event.event_type in {EventType.TASK_STARTED, EventType.TASK_SUCCEEDED}
    ]


def test_arm1_runs_two_base_microbatch_then_finishes_one_fin_tray() -> None:
    runtime, snapshot = _run_three_order_mix()
    tasks = {item["task_id"]: item for item in snapshot["tasks"]}
    arm1_events = _arm1_events(runtime)

    first_prepare = next(
        event
        for event in arm1_events
        if event.event_type is EventType.TASK_STARTED
        and event.payload["task_type"] == TaskType.PREPARE_FIN_TOOL.value
    )
    bases_before_prepare = [
        event
        for event in arm1_events
        if event.event_type is EventType.TASK_SUCCEEDED
        and event.payload["task_type"] == TaskType.PLACE_BASE_PLATE.value
        and event.sim_time <= first_prepare.sim_time
    ]
    assert len(bases_before_prepare) == 2

    first_fin_unit = tasks[first_prepare.payload["task_id"]]["unit_id"]
    first_unit_install_finishes = [
        event.sim_time
        for event in arm1_events
        if event.event_type is EventType.TASK_SUCCEEDED
        and event.payload["task_type"] == TaskType.INSTALL_FIN.value
        and tasks[event.payload["task_id"]]["unit_id"] == first_fin_unit
    ]
    assert first_unit_install_finishes

    later_base_starts = [
        event.sim_time
        for event in arm1_events
        if event.event_type is EventType.TASK_STARTED
        and event.payload["task_type"] == TaskType.PICK_BASE_PLATE.value
        and event.sim_time > first_prepare.sim_time
    ]
    assert later_base_starts
    assert max(first_unit_install_finishes) <= min(later_base_starts)


def test_soft_microbatch_continues_base_work_when_no_fin_action_is_imminent() -> None:
    runtime, _snapshot = _run_three_order_mix(
        Arm1ToolPolicyConfig(
            max_base_microbatch=2,
            lookahead_seconds=10.0,
            starvation_seconds=30.0,
        )
    )
    arm1_events = _arm1_events(runtime)
    first_prepare = next(
        event
        for event in arm1_events
        if event.event_type is EventType.TASK_STARTED
        and event.payload["task_type"] == TaskType.PREPARE_FIN_TOOL.value
    )
    bases_before_prepare = [
        event
        for event in arm1_events
        if event.event_type is EventType.TASK_SUCCEEDED
        and event.payload["task_type"] == TaskType.PLACE_BASE_PLATE.value
        and event.sim_time <= first_prepare.sim_time
    ]

    assert len(bases_before_prepare) == 3


def test_default_policy_reduces_line_completion_time_and_arm1_idle_time() -> None:
    _baseline_runtime, baseline = _run_three_order_mix(
        Arm1ToolPolicyConfig(
            max_base_microbatch=999,
            lookahead_seconds=12.0,
            starvation_seconds=30.0,
        )
    )
    _optimized_runtime, optimized = _run_three_order_mix()

    baseline_parallelism = baseline["async_line"]["parallelism"]
    optimized_parallelism = optimized["async_line"]["parallelism"]
    assert optimized["terminal"] is True
    assert all(order["status"] == "COMPLETED" for order in optimized["orders"])
    assert optimized["elapsed"] < baseline["elapsed"]
    assert optimized_parallelism["arm1_idle_s"] < baseline_parallelism["arm1_idle_s"]
    assert optimized_parallelism["arm1_utilization"] > baseline_parallelism["arm1_utilization"]


def test_v2_releases_next_wave_base_work_before_the_first_furnace_batch_finishes() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.set_execution_gate(_InstantPhysicalGate())
    first_wave = {"WIP_1", "WIP_2", "WIP_3"}
    next_wave = {"WIP_4", "WIP_5"}
    for index, preset in enumerate(("A", "B", "C", "D", "A"), start=1):
        runtime.submit_order(preset, order_id=f"WIP_{index}")

    for _index in range(5_000):
        runtime.tick(0.05)
        if runtime.complete:
            break
    else:
        raise AssertionError("five-order unified V2 scenario did not complete")

    tasks = runtime.manufacturing_runtime.graph.tasks
    arm1_events = [event for event in runtime.manufacturing_runtime.events.history if event.source == "ARM1"]
    next_wave_base_started = min(
        event.sim_time
        for event in arm1_events
        if event.event_type is EventType.TASK_STARTED
        and tasks[event.payload["task_id"]].order_id in next_wave
        and tasks[event.payload["task_id"]].task_type is TaskType.PICK_BASE_PLATE
    )
    preceding_first_wave_arm1_finish = max(
        event.sim_time
        for event in arm1_events
        if event.event_type is EventType.TASK_SUCCEEDED
        and event.sim_time <= next_wave_base_started
        and tasks[event.payload["task_id"]].order_id in first_wave
    )
    first_wave_delivered = min(
        float(event["time"])
        for event in runtime.physical_runtime.events
        if event["type"] == "UNIT_COMPLETED" and str(event["order_id"]) in first_wave
    )

    assert next_wave_base_started - preceding_first_wave_arm1_finish <= 1.0
    assert next_wave_base_started < first_wave_delivered


def test_v2_deferred_tool_control_node_does_not_claim_arm1_fin_residency() -> None:
    policy = Arm1ToolResidencyPolicy(
        Arm1ToolPolicyConfig(),
        initial_tool="vacuum_gripper",
    )
    task = ManufacturingTask(
        task_id="V2_DEFERRED_PREPARE",
        task_type=TaskType.PREPARE_FIN_TOOL,
        order_id="V2_DEFERRED",
        unit_id="V2_DEFERRED_UNIT_01",
        eligible_resources=["ARM1"],
        successors=["V2_DEFERRED_PICK_FIN"],
        payload={"arm1_tool_policy_neutral": True},
    )

    policy.observe_started(task, "ARM1")

    assert policy.current_tool == "vacuum_gripper"
    assert policy.active_fin_unit is None
