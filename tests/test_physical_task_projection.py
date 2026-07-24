from __future__ import annotations

from typing import Any

from brazing_sim.physical_task_projection import PhysicalTaskStatusProjector


def task(
    task_id: str,
    task_type: str,
    *,
    predecessors: tuple[str, ...] = (),
    unit: int = 1,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "order_id": "ORDER_001",
        "unit_id": f"ORDER_001_UNIT_{unit:02d}",
        "tray_id": f"tray_{unit:02d}",
        "predecessors": list(predecessors),
        "successors": [],
        "payload": dict(payload or {}),
        # Deliberately simulate the old duration-only scheduler racing ahead.
        "status": "SUCCEEDED",
    }


def physical_state() -> dict[str, Any]:
    return {
        "order_id": "ORDER_001",
        "stage": "BASE_LOADING",
        "fixture": {},
        "fins": {},
        "paths": {},
        "arms": {
            "arm1": {"task_type": "", "status": "idle"},
            "arm2": {"task_type": "", "status": "idle"},
            "arm3": {"task_type": "", "status": "idle"},
        },
        "tools": {"arm1": {"current_tool": "suction_tool"}},
    }


def statuses(items: list[dict[str, Any]]) -> dict[str, str]:
    return {str(item["task_id"]): str(item["status"]) for item in items}


def test_scheduler_completion_cannot_turn_node_green_before_physical_completion() -> None:
    projector = PhysicalTaskStatusProjector()
    tasks = [
        task("pick", "PICK_BASE_PLATE"),
        task("place", "PLACE_BASE_PLATE", predecessors=("pick",)),
    ]

    projected = projector.project(tasks, physical_state())

    assert statuses(projected) == {"pick": "READY", "place": "PENDING"}
    assert all(item["scheduler_status"] == "SUCCEEDED" for item in projected)
    assert all(item["status_source"] == "PHYSICAL" for item in projected)


def test_base_pick_and_place_turn_green_at_their_own_physical_milestones() -> None:
    projector = PhysicalTaskStatusProjector()
    tasks = [
        task("pick", "PICK_BASE_PLATE"),
        task("place", "PLACE_BASE_PLATE", predecessors=("pick",)),
    ]
    physical = physical_state()
    physical["arms"]["arm1"]["task_type"] = "LOAD_BASE"

    assert statuses(projector.project(tasks, physical)) == {
        "pick": "RUNNING",
        "place": "PENDING",
    }

    picked = projector.project(
        tasks,
        physical,
        active_task_type="LOAD_BASE",
        active_task_payload={"physical_pick_complete": True},
    )
    assert statuses(picked) == {"pick": "SUCCEEDED", "place": "RUNNING"}

    placed = projector.project(
        tasks,
        physical,
        active_task_type="LOAD_BASE",
        active_task_payload={
            "physical_pick_complete": True,
            "physical_place_complete": True,
        },
    )
    assert statuses(placed) == {"pick": "SUCCEEDED", "place": "SUCCEEDED"}

    physical["arms"]["arm1"]["task_type"] = ""
    physical["fixture"]["base_weld_active"] = True
    assert statuses(projector.project(tasks, physical)) == {
        "pick": "SUCCEEDED",
        "place": "SUCCEEDED",
    }


def test_fin_pick_and_install_turn_green_at_their_own_physical_milestones() -> None:
    projector = PhysicalTaskStatusProjector()
    tasks = [
        task("pick_fin", "PICK_FIN", payload={"fin_id": "fin_01"}),
        task(
            "install_fin",
            "INSTALL_FIN",
            predecessors=("pick_fin",),
            payload={"fin_id": "fin_01"},
        ),
    ]
    physical = physical_state()
    physical["stage"] = "FIN_ASSEMBLY"
    physical["fins"] = {"fin_01": {"fin_id": "fin_01", "active": True, "inserted": False}}

    moving = projector.project(
        tasks,
        physical,
        active_task_type="INSERT_FIN",
        active_task_payload={"fin_id": "fin_01"},
    )
    assert statuses(moving) == {"pick_fin": "RUNNING", "install_fin": "PENDING"}

    picked = projector.project(
        tasks,
        physical,
        active_task_type="INSERT_FIN",
        active_task_payload={"fin_id": "fin_01", "physical_pick_complete": True},
    )
    assert statuses(picked) == {"pick_fin": "SUCCEEDED", "install_fin": "RUNNING"}

    placed = projector.project(
        tasks,
        physical,
        active_task_type="INSERT_FIN",
        active_task_payload={
            "fin_id": "fin_01",
            "physical_pick_complete": True,
            "physical_place_complete": True,
        },
    )
    assert statuses(placed) == {"pick_fin": "SUCCEEDED", "install_fin": "SUCCEEDED"}

    physical["fins"]["fin_01"]["inserted"] = True
    done = projector.project(tasks, physical)
    assert statuses(done) == {"pick_fin": "SUCCEEDED", "install_fin": "SUCCEEDED"}


def test_rack_nodes_follow_real_transfer_steps_instead_of_estimated_time() -> None:
    projector = PhysicalTaskStatusProjector()
    tasks = [
        task("out", "TRANSFER_TRAY_OUT"),
        task("lift", "MOVE_ELEVATOR", predecessors=("out",), payload={"layer_index": 0}),
        task("load", "LOAD_RACK_LAYER", predecessors=("lift",), payload={"layer_index": 0}),
        task("lock", "LOCK_RACK_LAYER", predecessors=("load",), payload={"layer_index": 0}),
    ]
    physical = physical_state()
    physical.update(
        {
            "stage": "TRANSFERRING_LAYER",
            "batch": {
                "stage": "TRANSFERRING_LAYER",
                "units": [{"unit_id": "tray_01", "phase": "TRANSFERRING"}],
            },
            "transfer": {"unit_id": "tray_01", "step": "load_push"},
        }
    )

    assert statuses(projector.project(tasks, physical)) == {
        "out": "SUCCEEDED",
        "lift": "SUCCEEDED",
        "load": "RUNNING",
        "lock": "PENDING",
    }

    physical["batch"]["units"][0]["phase"] = "LOCKED"
    physical["transfer"]["step"] = "load_retract"
    assert all(status == "SUCCEEDED" for status in statuses(projector.project(tasks, physical)).values())


def test_finished_route_waits_until_output_gate_handoff_is_complete() -> None:
    projector = PhysicalTaskStatusProjector()
    tasks = [
        task("inspect", "POST_BRAZE_INSPECTION"),
        task(
            "route",
            "ROUTE_PASS",
            predecessors=("inspect",),
            payload={"condition": "PASS"},
        ),
    ]
    physical = physical_state()
    physical.update(
        {
            "stage": "POST_INSPECTION",
            "batch": {
                "stage": "POST_INSPECTION",
                "units": [
                    {
                        "unit_id": "tray_01",
                        "phase": "INSPECTED",
                        "disposition": "PASS",
                    }
                ],
            },
            "transfer": {"unit_id": "tray_01", "step": "delivery_gate_open"},
        }
    )

    assert statuses(projector.project(tasks, physical)) == {
        "inspect": "SUCCEEDED",
        "route": "RUNNING",
    }

    physical["batch"]["units"][0]["phase"] = "DELIVERED"
    physical["transfer"]["step"] = ""
    assert statuses(projector.project(tasks, physical)) == {
        "inspect": "SUCCEEDED",
        "route": "SUCCEEDED",
    }
