from __future__ import annotations

from dataclasses import replace

from brazing_sim.events import EventType
from brazing_sim.flexible import build_inline_plan
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.manufacturing_config import BatchingConfig, load_scheduler_config
from brazing_sim.planning.task_models import TaskStatus, TaskType
from brazing_sim.recovery import FaultType
from brazing_sim.workcells import TrayRoutePhase


def _runtime_with_batch_wait(max_wait_time: float) -> ManufacturingRuntime:
    config = load_scheduler_config("config/scheduler.yaml")
    config = replace(
        config,
        batching=BatchingConfig(
            mode=config.batching.mode,
            max_wait_time=max_wait_time,
            allow_partial_batch=config.batching.allow_partial_batch,
            maximum_units=config.batching.maximum_units,
        ),
    )
    return ManufacturingRuntime(
        scheduler_mode="dynamic",
        scheduler_config=config,
        flexible_cell=True,
    )


def _run(runtime: ManufacturingRuntime, *, limit: int = 12000, step: float = 0.25) -> float:
    start = -step if runtime.last_tick is None else float(runtime.last_tick)
    for index in range(limit):
        now = start + (index + 1) * step
        runtime.tick(now)
        if runtime.terminal:
            return now
    raise AssertionError("runtime did not terminate")


def test_compatible_orders_share_one_runtime_furnace_cycle() -> None:
    runtime = _runtime_with_batch_wait(500.0)
    runtime.submit_plan(
        build_inline_plan(preset="A", order_id="SHARED_BATCH_A", quantity=1, priority=10),
        now=0.0,
    )
    runtime.submit_plan(
        build_inline_plan(preset="B", order_id="SHARED_BATCH_B", quantity=1, priority=10),
        now=0.0,
    )

    _run(runtime)

    furnace_starts = [
        event
        for event in runtime.events.history
        if event.event_type is EventType.TASK_STARTED
        and event.payload.get("task_type") == TaskType.RUN_FURNACE.value
    ]
    assert len(furnace_starts) == 1
    batches = runtime.snapshot()["furnace_batches"]
    assert len(batches) == 1
    assert batches[0]["unit_count"] == 2
    assert batches[0]["order_ids"] == ["SHARED_BATCH_A", "SHARED_BATCH_B"]
    assert batches[0]["status"] == "COMPLETED"
    unload_layers = [
        int(runtime.graph.get(event.payload["task_id"]).payload["layer_index"])
        for event in runtime.events.history
        if event.event_type is EventType.TASK_STARTED
        and event.payload.get("task_type") == TaskType.UNLOAD_RACK_LAYER.value
    ]
    assert unload_layers == sorted(unload_layers, reverse=True)


def test_three_units_from_one_order_keep_one_full_furnace_cycle() -> None:
    runtime = _runtime_with_batch_wait(30.0)
    runtime.submit_plan(
        build_inline_plan(preset="A", order_id="FULL_ORDER_BATCH", quantity=3, priority=10),
        now=0.0,
    )

    _run(runtime)

    furnace_starts = [
        event
        for event in runtime.events.history
        if event.event_type is EventType.TASK_STARTED
        and event.payload.get("task_type") == TaskType.RUN_FURNACE.value
    ]
    assert len(furnace_starts) == 1
    batch = runtime.snapshot()["furnace_batches"][0]
    assert batch["unit_count"] == 3
    assert batch["order_ids"] == ["FULL_ORDER_BATCH"]


def test_incompatible_orders_never_share_a_furnace_cycle() -> None:
    runtime = _runtime_with_batch_wait(500.0)
    first = build_inline_plan(
        preset="A",
        order_id="INCOMPATIBLE_A",
        quantity=1,
        priority=10,
    )
    second = build_inline_plan(
        preset="B",
        order_id="INCOMPATIBLE_B",
        quantity=1,
        priority=10,
    )
    second = replace(second, recipe=replace(second.recipe, name="other_recipe"))
    runtime.submit_plan(first, now=0.0)
    runtime.submit_plan(second, now=0.0)

    _run(runtime)

    batches = runtime.snapshot()["furnace_batches"]
    assert len(batches) == 2
    assert all(batch["unit_count"] == 1 for batch in batches)
    assert {tuple(batch["order_ids"]) for batch in batches} == {
        ("INCOMPATIBLE_A",),
        ("INCOMPATIBLE_B",),
    }


def test_shared_furnace_interlock_recovery_releases_every_batch_member() -> None:
    runtime = _runtime_with_batch_wait(30.0)
    entry = runtime.submit_plan(
        build_inline_plan(preset="A", order_id="RECOVER_SHARED_BATCH", quantity=3, priority=10),
        now=0.0,
    )
    injected = False
    for index in range(12000):
        now = index * 0.25
        runtime.tick(now)
        running = next(
            (
                task
                for task in runtime.graph
                if task.task_type is TaskType.RUN_FURNACE and task.status is TaskStatus.RUNNING
            ),
            None,
        )
        if running is not None and not injected:
            runtime.inject_fault(
                FaultType.FURNACE_DOOR_INTERLOCK,
                source="FURNACE",
                related_task_id=running.task_id,
                now=now,
            )
            injected = True
        if runtime.terminal:
            break

    assert injected
    assert runtime.terminal
    assert entry.status.value == "COMPLETED"
    batch = runtime.snapshot()["furnace_batches"][0]
    assert batch["status"] == "COMPLETED"
    assert batch["unit_count"] == 3


def test_completed_unit_releases_wip_before_its_order_finishes() -> None:
    runtime = _runtime_with_batch_wait(30.0)
    first = runtime.submit_plan(
        build_inline_plan(preset="A", order_id="WIP_RELEASE_Q3", quantity=3, priority=10),
        now=0.0,
    )
    waiting = runtime.submit_plan(
        build_inline_plan(preset="B", order_id="WIP_RELEASE_NEXT", quantity=1, priority=20),
        now=0.0,
    )
    assert waiting.status.value == "QUEUED"

    observation = None
    for index in range(12000):
        now = index * 0.25
        runtime.tick(now)
        delivered = [
            task
            for task in runtime.graph
            if task.order_id == first.order_id
            and task.task_type in {TaskType.ROUTE_PASS, TaskType.ROUTE_REWORK, TaskType.ROUTE_SCRAP}
            and task.status is TaskStatus.SUCCEEDED
        ]
        if delivered and first.status.value != "COMPLETED":
            observation = {
                "waiting_status": waiting.status.value,
                "waiting_trays": dict(waiting.tray_assignments),
                "active_wip": runtime.snapshot(now)["async_line"]["active_wip"],
            }
            break

    assert observation is not None
    assert observation["waiting_status"] in {"RELEASED", "RUNNING"}
    assert observation["waiting_trays"]
    assert observation["active_wip"] == 3


def test_multi_unit_order_is_admitted_one_tray_at_a_time() -> None:
    runtime = _runtime_with_batch_wait(30.0)
    first = runtime.submit_plan(
        build_inline_plan(preset="A", order_id="WIP_HOLDER_Q3", quantity=3, priority=10),
        now=0.0,
    )
    waiting = runtime.submit_plan(
        build_inline_plan(preset="B", order_id="WIP_PARTIAL_Q2", quantity=2, priority=20),
        now=0.0,
    )

    observation = None
    for index in range(12000):
        now = index * 0.25
        runtime.tick(now)
        delivered = [
            task
            for task in runtime.graph
            if task.order_id == first.order_id
            and task.task_type in {TaskType.ROUTE_PASS, TaskType.ROUTE_REWORK, TaskType.ROUTE_SCRAP}
            and task.status is TaskStatus.SUCCEEDED
        ]
        if delivered and first.status.value != "COMPLETED":
            observation = {
                "status": waiting.status.value,
                "assigned_count": len(waiting.tray_assignments),
                "active_wip": runtime.snapshot(now)["async_line"]["active_wip"],
            }
            break

    assert observation == {
        "status": "RELEASED",
        "assigned_count": 1,
        "active_wip": 3,
    }

    _run(runtime)
    assert waiting.status.value == "COMPLETED"
    assert len(waiting.tray_assignments) == 2


def test_fourth_unit_enters_upstream_wip_without_waiting_for_a_furnace_layer() -> None:
    runtime = ManufacturingRuntime(
        scheduler_mode="dynamic",
        flexible_cell=True,
        max_wip_units=6,
    )
    entries = []
    for index, preset in enumerate(("A", "B", "C", "A"), start=1):
        entry = runtime.submit_plan(
            build_inline_plan(
                preset=preset,
                order_id=f"LAYER_QUEUE_{index}_{preset}",
                quantity=1,
                priority=10,
            ),
            now=0.0,
        )
        entries.append(entry)

    assert entries[-1].status.value == "RELEASED"
    assert entries[-1].admitted_unit_ids == {"LAYER_QUEUE_4_A_UNIT_01"}
    assert len(runtime.tray_routes) == 6
    assert runtime._active_wip() == 4
    assert entries[-1].status.value in {"RELEASED", "RUNNING", "COMPLETED"}
    _run(runtime)
    assert all(entry.status.value == "COMPLETED" for entry in entries)


def test_v2_projection_advances_across_omitted_visual_route_phases() -> None:
    runtime = ManufacturingRuntime(scheduler_mode="dynamic", flexible_cell=False)
    entry = runtime.submit_plan(
        build_inline_plan(
            preset="A",
            order_id="ROUTE_PROJECTION",
            quantity=1,
            priority=10,
        ),
        now=0.0,
    )
    lock_task = next(
        runtime.graph.get(task_id)
        for task_id in entry.graph_task_ids
        if runtime.graph.get(task_id).task_type is TaskType.LOCK_RACK_LAYER
    )

    runtime._advance_cell_state(lock_task, 1.0)

    route = runtime.tray_routes[lock_task.tray_id]
    assert route.phase is TrayRoutePhase.FURNACE
    assert runtime.last_error == ""
