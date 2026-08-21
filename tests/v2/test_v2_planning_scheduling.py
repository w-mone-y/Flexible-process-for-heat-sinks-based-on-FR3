from __future__ import annotations

import pytest

from brazing_sim.flexible import build_preset_plan
from brazing_sim.planning import TaskGraph, TaskGraphError, build_task_graph
from brazing_sim.planning.batch_planner import BatchCandidate, BatchPlanner, are_units_batch_compatible
from brazing_sim.planning.task_models import (
    TASK_STATUS_LABELS_ZH,
    TASK_TYPE_LABELS_ZH,
    ManufacturingTask,
    TaskStatus,
    TaskType,
)
from brazing_sim.scheduling import (
    DynamicPriorityScheduler,
    FixedSequenceScheduler,
    ResourceState,
    ResourceStatus,
    ZoneLockManager,
)


def task(task_id: str, *, resource: str = "ARM1", priority: int = 0, zone: str = ""):
    return ManufacturingTask(
        task_id=task_id,
        task_type=TaskType.PICK_FIN,
        order_id="O",
        unit_id="U",
        eligible_resources=[resource],
        required_zones=[] if not zone else [zone],
        priority=priority,
    )


def test_process_plan_builds_valid_parallel_dag() -> None:
    graph = build_task_graph(build_preset_plan("A", quantity=1))
    graph.validate_acyclic()
    ready_types = {item.task_type for item in graph.get_ready_tasks()}
    assert ready_types == {TaskType.PICK_BASE_PLATE}
    place = next(item for item in graph if item.task_type is TaskType.PLACE_BASE_PLATE)
    graph.get(place.predecessors[0]).status = TaskStatus.SUCCEEDED
    graph.refresh_ready(1.0)
    assert place.status is TaskStatus.READY
    place.status = TaskStatus.SUCCEEDED
    graph.refresh_ready(2.0)
    assert {item.task_type for item in graph.get_ready_tasks()} >= {
        TaskType.PREPARE_FIN_TOOL,
        TaskType.DISPENSE_BRAZING,
    }


def test_every_task_node_has_payload_aware_chinese_text() -> None:
    graph = build_task_graph(build_preset_plan("A", quantity=1))
    assert set(TASK_TYPE_LABELS_ZH) == set(TaskType)
    assert set(TASK_STATUS_LABELS_ZH) == set(TaskStatus)
    snapshots = {item["task_type"]: item for item in graph.snapshot()}
    assert snapshots[TaskType.PICK_BASE_PLATE.value]["display_name_zh"] == "吸取基板"
    assert snapshots[TaskType.PICK_FIN.value]["display_detail_zh"] == "工件1 · 翅片5"
    assert snapshots[TaskType.DISPENSE_BRAZING.value]["display_detail_zh"] == "工件1 · 共10条钎料路径"
    assert snapshots[TaskType.LOAD_RACK_LAYER.value]["display_detail_zh"] == "工件1 · 第1层"
    assert snapshots[TaskType.RUN_FURNACE.value]["display_detail_zh"] == "钎焊周期10秒"
    assert snapshots[TaskType.ROUTE_PASS.value]["display_name_zh"] == "合格品分流"
    assert all(item["status_zh"] in TASK_STATUS_LABELS_ZH.values() for item in snapshots.values())


def test_cycle_is_rejected_and_failure_blocks_descendant() -> None:
    graph = TaskGraph()
    first = task("first")
    second = task("second")
    second.predecessors = ["first"]
    graph.add_task(first)
    graph.add_task(second)
    with pytest.raises(TaskGraphError):
        graph.add_dependency("second", "first")
    graph.refresh_ready(0.0)
    graph.mark_failed("first", "test", 1.0)
    assert second.status is TaskStatus.BLOCKED


def test_topological_order_cache_is_invalidated_by_dependency_changes() -> None:
    graph = TaskGraph((task("a"), task("b")))
    assert [item.task_id for item in graph.topological_order()] == ["a", "b"]

    graph.add_dependency("b", "a")

    assert [item.task_id for item in graph.topological_order()] == ["b", "a"]


def test_zone_lock_is_atomic_and_released_by_task() -> None:
    locks = ZoneLockManager(["ZONE_A", "ZONE_B"])
    assert locks.acquire("t1", "ARM1", ["ZONE_A", "ZONE_B"], 0.0)
    assert not locks.acquire("t2", "ARM2", ["ZONE_B"], 0.1)
    assert locks.release("t1") == 2
    assert locks.acquire("t2", "ARM2", ["ZONE_B"], 0.2)


def test_dynamic_scheduler_respects_fault_tool_zone_and_priority() -> None:
    low = task("low", resource="ARM1", priority=1, zone="ZONE_A")
    high = task("high", resource="ARM2", priority=20, zone="ZONE_B")
    for item in (low, high):
        item.status = TaskStatus.READY
    resources = {
        "ARM1": ResourceState("ARM1", "ROBOT", capabilities={TaskType.PICK_FIN.value}, available_tools=set()),
        "ARM2": ResourceState("ARM2", "ROBOT", capabilities={TaskType.PICK_FIN.value}, available_tools=set()),
    }
    scheduler = DynamicPriorityScheduler(max_assignments_per_tick=2)
    assignments = scheduler.select_assignments((low, high), resources, {"occupied_zones": set()}, 0.0)
    assert [item.task_id for item in assignments][0] == "high"
    resources["ARM2"].status = ResourceStatus.FAULTED
    assignments = scheduler.select_assignments((low, high), resources, {"occupied_zones": set()}, 0.0)
    assert [item.task_id for item in assignments] == ["low"]
    assert not scheduler.select_assignments((low,), resources, {"occupied_zones": {"ZONE_A"}}, 0.0)


def test_dynamic_scheduler_uses_arm3_detection_priority_at_fin_boundaries() -> None:
    inspect = ManufacturingTask(
        task_id="inspect_unlocks_arm1",
        task_type=TaskType.INSPECT_BRAZING,
        order_id="INSPECT_ORDER",
        unit_id="INSPECT_UNIT",
        eligible_resources=["ARM3"],
        estimated_duration=10.0,
        priority=1,
        status=TaskStatus.READY,
    )
    next_fin = ManufacturingTask(
        task_id="continue_arm3_fin",
        task_type=TaskType.INSTALL_FIN,
        order_id="FIN_ORDER",
        unit_id="FIN_UNIT",
        eligible_resources=["ARM3"],
        estimated_duration=1.0,
        priority=100,
        status=TaskStatus.READY,
    )
    resources = {
        "ARM3": ResourceState(
            "ARM3",
            "ROBOT",
            capabilities={TaskType.INSPECT_BRAZING.value, TaskType.INSTALL_FIN.value},
            available_tools=set(),
        )
    }
    scheduler = DynamicPriorityScheduler(max_assignments_per_tick=1)

    assignments = scheduler.select_assignments(
        (next_fin, inspect),
        resources,
        {
            "occupied_zones": set(),
            "resource_task_type_priorities": {
                "ARM3": (
                    (TaskType.INSPECT_BRAZING.value, TaskType.INSPECT_FINS.value),
                    (TaskType.PICK_FIN.value, TaskType.INSTALL_FIN.value),
                )
            },
        },
        0.0,
    )

    assert [assignment.task_id for assignment in assignments] == [inspect.task_id]


def test_fixed_scheduler_is_deterministic() -> None:
    values = [task("b"), task("a")]
    for index, item in enumerate(values):
        item.status = TaskStatus.READY
        item.sequence_index = index
    resources = {"ARM1": ResourceState("ARM1", "ROBOT", capabilities={TaskType.PICK_FIN.value})}
    selected = FixedSequenceScheduler().select_assignments(values, resources, {}, 0.0)
    assert selected[0].task_id == "b"


def test_batch_compatibility_checks_recipe_material_and_capacity() -> None:
    a = build_preset_plan("A", quantity=1)
    b = build_preset_plan("B", quantity=1)
    assert are_units_batch_compatible((a, b))
    assert not are_units_batch_compatible((build_preset_plan("A", quantity=3), b))
    planner = BatchPlanner(max_wait_time=30.0)
    assert not planner.select_batch((BatchCandidate(a, 0.0),), 29.9)
    assert planner.select_batch((BatchCandidate(a, 0.0),), 30.0) == (a,)
