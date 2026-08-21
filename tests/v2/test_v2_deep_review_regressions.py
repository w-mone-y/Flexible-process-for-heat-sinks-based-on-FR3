from __future__ import annotations

from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from brazing_sim.api import SharedState, start_http_server, validate_http_command
from brazing_sim.dual_line import DualLineRuntime, UnitStage
from brazing_sim.dual_line.runtime import EVENT_HISTORY_LIMIT
from brazing_sim.dual_line.unified_runtime import V2PhysicalExecutionBridge
from brazing_sim.planning import ManufacturingTask, TaskType


def _task(task_id: str, task_type: TaskType, unit_id: str, **payload: object) -> ManufacturingTask:
    return ManufacturingTask(
        task_id=task_id,
        task_type=task_type,
        order_id="REVIEW_ORDER",
        unit_id=unit_id,
        eligible_resources=["ARM2" if task_type is TaskType.REWORK_BRAZING else "ARM3"],
        payload=dict(payload),
    )


def test_physical_recovery_tasks_need_fresh_measured_events() -> None:
    runtime = DualLineRuntime(fast=True)
    order = runtime.submit_order("A", order_id="RECOVERY_EVENT_BOUNDARY")
    unit_id = order.unit_ids[0]
    bridge = V2PhysicalExecutionBridge(runtime)

    rework = _task("REWORK", TaskType.REWORK_BRAZING, unit_id)
    bridge.authorize(rework, "ARM2")
    assert not bridge.task_complete(rework, "ARM2", since=0.0)[0]

    runtime.sim_time = 1.0
    runtime._event("OPERATION_COMPLETED", unit_id=unit_id, kind="DISPENSING")
    assert bridge.task_complete(rework, "ARM2", since=1.0)[0]

    reinstall = _task("REINSTALL", TaskType.REINSTALL_FIN, unit_id, fin_id="fin_02")
    bridge.authorize(reinstall, "ARM3")
    runtime.sim_time = 2.0
    runtime._event("FIN_INSTALLED", unit_id=unit_id, fin_index=2)
    assert not bridge.task_complete(reinstall, "ARM3", since=3.0)[0]
    runtime.sim_time = 4.0
    runtime._event("FIN_INSTALLED", unit_id=unit_id, fin_index=2)
    assert bridge.task_complete(reinstall, "ARM3", since=3.0)[0]


def test_split_physical_milestones_release_their_shared_permits() -> None:
    runtime = DualLineRuntime(fast=True)
    order = runtime.submit_order("A", order_id="PERMIT_RELEASE")
    unit = runtime.units[order.unit_ids[0]]
    bridge = V2PhysicalExecutionBridge(runtime)

    place = _task("PLACE", TaskType.PLACE_BASE_PLATE, unit.unit_id)
    unit.stage = UnitStage.WAITING_S2A
    bridge.authorize(place, "ARM3")
    assert bridge.task_complete(place, "ARM3", since=10.0)[0]
    bridge.revoke_completed_permits(place, since=10.0)
    assert (unit.unit_id, "BASE_LOADING") not in bridge._permits

    install = _task("INSTALL", TaskType.INSTALL_FIN, unit.unit_id, fin_id="fin_01")
    unit.stage = UnitStage.FIN_INSTALLATION
    unit.fins_installed = 1
    bridge.authorize(install, "ARM3")
    assert bridge.task_complete(install, "ARM3", since=20.0)[0]
    bridge.revoke_completed_permits(install, since=20.0)
    assert (unit.unit_id, "INSTALL_FIN") not in bridge._permits


def test_http_order_boundary_rejects_coercion_and_oversized_ids() -> None:
    base = {"line_profile": "V2_DUAL_INSTALL", "mode": "preset", "preset": "A"}
    with pytest.raises(ValueError, match="quantity.*整数"):
        validate_http_command("/orders/plan", {**base, "quantity": 1.5})
    with pytest.raises(ValueError, match="priority.*整数"):
        validate_http_command("/orders/plan", {**base, "priority": True})
    with pytest.raises(ValueError, match="长度"):
        validate_http_command("/orders/plan", {**base, "order_id": "x" * 129})
    with pytest.raises(ValueError, match="不支持首选料架层"):
        validate_http_command(
            "/orders/insert",
            {**base, "order_id": "NO_LAYER", "preferred_rack_layer": 2},
        )


def test_remote_http_control_requires_bearer_token() -> None:
    shared = SharedState()
    with pytest.raises(ValueError, match="remote HTTP control"):
        start_http_server(shared, "0.0.0.0", 0)

    server = start_http_server(shared, "127.0.0.1", 0, auth_token="review-secret")
    url = f"http://127.0.0.1:{server.server_address[1]}/stop"
    try:
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(Request(url, data=b"{}", method="POST"), timeout=2.0)
        assert unauthorized.value.code == 401

        request = Request(
            url,
            data=b"{}",
            headers={"Authorization": "Bearer review-secret"},
            method="POST",
        )
        with urlopen(request, timeout=2.0) as response:
            assert response.status == 202
        assert shared.commands.get_nowait()["type"] == "stop"
    finally:
        server.shutdown()
        server.server_close()


def test_v2_runtime_event_history_is_bounded() -> None:
    runtime = DualLineRuntime(fast=True)
    for index in range(EVENT_HISTORY_LIMIT + 25):
        runtime._event("REVIEW_EVENT", index=index)
    assert len(runtime.events) == EVENT_HISTORY_LIMIT
    assert runtime.events[0]["index"] == 25
