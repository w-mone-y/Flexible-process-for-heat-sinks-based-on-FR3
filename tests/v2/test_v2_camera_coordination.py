from __future__ import annotations

from dataclasses import dataclass

from brazing_sim.dual_line import DualLineRuntime, UnifiedV2Runtime
from brazing_sim.dual_line.application import V2BrazingApplication, V2ControlSurface
from brazing_sim.dual_line.cli import parse_args
from brazing_sim.dual_line.tray_flow import TrayOwner
from brazing_sim.flexible import build_inline_plan
from brazing_sim.planning import TaskType

INSPECTION_KINDS = {"MATERIAL_INSPECTION", "PRE_BRAZE_INSPECTION"}


@dataclass
class _InstantPhysicalGate:
    """Minimal measured gate for exercising UnifiedV2Runtime without MuJoCo."""

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


def _run(runtime: DualLineRuntime, *, limit: int = 12_000) -> dict[str, object]:
    for _index in range(limit):
        runtime.tick(0.02)
        if runtime.complete:
            return runtime.snapshot()
    raise AssertionError("V2 camera-coordination scenario did not complete")


def _plan(order_id: str, strategy: str, *, quantity: int = 1):
    return build_inline_plan(
        preset="A",
        order_id=order_id,
        quantity=quantity,
        priority=20,
        route_strategy=strategy,
    )


def test_standard_route_uses_arm3_camera_at_s2b_and_s4_without_redundant_review() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="CAMERA_STANDARD")

    snapshot = _run(runtime)

    starts = [event for event in snapshot["events"] if event["type"] == "OPERATION_STARTED"]
    assert {(event["resource"], event["kind"]) for event in starts if event["kind"] in INSPECTION_KINDS} == {
        ("ARM3", "MATERIAL_INSPECTION"),
        ("ARM3", "PRE_BRAZE_INSPECTION"),
    }
    assert snapshot["camera_coordination"]["review_history"] == []


def test_arm3_primary_result_enters_rework_without_queuing_a_redundant_closeup() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="CAMERA_SUSPICIOUS")
    runtime.inject_fault("BRAZING_MISSING", target="path_02")

    snapshot = _run(runtime)

    events = snapshot["events"]
    detected = next(event for event in events if event["type"] == "FAULT_DETECTED")
    returned = next(event for event in events if event["type"] == "RECOVERY_RETURN_STARTED")
    assert detected["time"] <= returned["time"]
    assert snapshot["camera_coordination"]["review_history"] == []


def test_repaired_product_requires_a_fresh_arm3_reinspection_at_s2b() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_order("A", order_id="CAMERA_REPAIR_SECOND_VIEW")
    runtime.inject_fault("BRAZING_MISSING", target="path_02")

    snapshot = _run(runtime)

    assert snapshot["camera_coordination"]["review_history"] == []
    primary_starts = [
        event
        for event in snapshot["events"]
        if event["type"] == "OPERATION_STARTED"
        and event["resource"] == "ARM3"
        and event["kind"] == "MATERIAL_INSPECTION"
    ]
    assert len(primary_starts) == 2


def test_high_reliability_route_runs_both_arm3_closeups_only_at_s3b() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_plan(_plan("CAMERA_HIGH_RELIABILITY", "HIGH_RELIABILITY"))

    observed_owners: list[TrayOwner] = []
    for _index in range(12_000):
        runtime.tick(0.02)
        operation = runtime.operations.get("ARM3")
        if (
            operation is not None
            and operation.kind in INSPECTION_KINDS
            and runtime.camera_review_required(operation.unit_id, operation.kind)
        ):
            unit = runtime.units[operation.unit_id]
            assert unit.tray_id is not None
            observed_owners.append(runtime.flow.get(unit.tray_id).owner)
        if runtime.complete:
            break
    else:
        raise AssertionError("S3B camera route did not complete")

    snapshot = runtime.snapshot()

    history = snapshot["camera_coordination"]["review_history"]
    assert {
        (item["inspection_kind"], item["reason"])
        for item in history
        if item["unit_id"] == "CAMERA_HIGH_RELIABILITY_UNIT_01"
    } >= {
        ("MATERIAL_INSPECTION", "HIGH_RELIABILITY"),
        ("PRE_BRAZE_INSPECTION", "HIGH_RELIABILITY"),
    }
    assert observed_owners
    assert set(observed_owners) == {TrayOwner.INSTALL_B}
    assert runtime.units["CAMERA_HIGH_RELIABILITY_UNIT_01"].branch.value == "ARM3_B"


def test_unified_high_reliability_dag_places_both_closeups_at_s3b() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.submit_plan(_plan("CAMERA_DAG_HIGH_RELIABILITY", "HIGH_RELIABILITY"))

    tasks = list(runtime.manufacturing_runtime.graph)
    by_type = {task.task_type: task for task in tasks}
    assert TaskType.REVIEW_BRAZING_CLOSEUP in by_type
    assert TaskType.REVIEW_FINS_CLOSEUP in by_type
    assert TaskType.VERIFY_BASE_ALIGNMENT not in by_type
    assert TaskType.SECOND_POST_BRAZE_VIEW not in by_type

    brazing = by_type[TaskType.INSPECT_BRAZING]
    configure = by_type[TaskType.CONFIGURE_COMB]
    brazing_review = by_type[TaskType.REVIEW_BRAZING_CLOSEUP]
    fins = by_type[TaskType.INSPECT_FINS]
    fins_review = by_type[TaskType.REVIEW_FINS_CLOSEUP]
    assert brazing.task_id in configure.predecessors
    assert brazing_review.predecessors == [configure.task_id]
    assert fins.task_id in fins_review.successors
    assert all(
        runtime.manufacturing_runtime.graph.get(task_id).task_type is TaskType.INSTALL_FIN
        for task_id in fins_review.predecessors
    )
    assert brazing_review.eligible_resources == ["ARM3"]
    assert fins_review.eligible_resources == ["ARM3"]
    assert brazing_review.station_id == "S3B_ARM3_INSTALL"
    assert fins_review.station_id == "S3B_ARM3_INSTALL"
    assert brazing_review.required_zones == ["ZONE_S3B_ARM3"]
    assert fins_review.required_zones == ["ZONE_S3B_ARM3"]


def test_unified_high_reliability_completes_both_physical_closeup_nodes() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.set_execution_gate(_InstantPhysicalGate())
    runtime.submit_plan(_plan("CAMERA_DAG_EXECUTION", "HIGH_RELIABILITY"))

    for _index in range(12_000):
        runtime.tick(0.02)
        if runtime.complete:
            break
    else:
        raise AssertionError("unified reliable route did not complete")

    review_tasks = [
        task
        for task in runtime.manufacturing_runtime.graph
        if task.task_type in {TaskType.REVIEW_BRAZING_CLOSEUP, TaskType.REVIEW_FINS_CLOSEUP}
    ]
    assert {task.assigned_resource for task in review_tasks} == {"ARM3"}
    assert all(task.status.value == "SUCCEEDED" for task in review_tasks)


def test_first_article_route_only_adds_closeups_to_the_first_unit() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_plan(_plan("CAMERA_FIRST_ARTICLE", "FIRST_ARTICLE", quantity=2))

    snapshot = _run(runtime)

    route_reviews = [
        item
        for item in snapshot["camera_coordination"]["review_history"]
        if item["reason"] == "FIRST_ARTICLE"
    ]
    assert route_reviews
    assert {item["unit_id"] for item in route_reviews} == {"CAMERA_FIRST_ARTICLE_UNIT_01"}


def test_unified_runtime_blocks_inspection_when_arm3_camera_is_offline() -> None:
    runtime = UnifiedV2Runtime(fast=True)
    runtime.set_execution_gate(_InstantPhysicalGate())
    runtime.physical_runtime.faults.isolated_resources["ARM3"] = None
    runtime.submit_order("A", order_id="CAMERA_UNIFIED_FALLBACK")

    for _index in range(2_000):
        runtime.tick(0.02)
        if (
            runtime.physical_runtime.units["CAMERA_UNIFIED_FALLBACK_UNIT_01"].stage.value
            == "MATERIAL_INSPECTION"
        ):
            break
    else:
        raise AssertionError("unit did not reach the offline S2B camera")

    starts = [
        event
        for event in runtime.snapshot()["events"]
        if event["type"] == "OPERATION_STARTED" and event["kind"] == "MATERIAL_INSPECTION"
    ]
    assert starts == []
    inspection_task = next(
        task for task in runtime.manufacturing_runtime.graph if task.task_type is TaskType.INSPECT_BRAZING
    )
    assert inspection_task.assigned_resource is None
    assert runtime.manufacturing_runtime.resources.get("ARM3").status.value == "OFFLINE"
    assert set(runtime.manufacturing_runtime.resources.states) >= {"ARM3"}
    assert not {
        "S2B_CAMERA",
        "S4_CAMERA",
    } & set(runtime.manufacturing_runtime.resources.states)
    assert runtime.snapshot()["camera_coordination"]["review_history"] == []


def test_arm3_camera_offline_blocks_at_s2b() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.faults.isolated_resources["ARM3"] = None
    runtime.submit_order("A", order_id="CAMERA_NO_FALLBACK")

    observed = None
    for _index in range(2_000):
        runtime.tick(0.02)
        station = next(
            item
            for item in runtime.camera_coordination_snapshot()["stations"]
            if item["station_id"] == "S2B_MATERIAL_INSPECTION"
        )
        if station["status"] == "BLOCKED":
            observed = station
            break

    assert observed is not None
    assert observed["status"] == "BLOCKED"
    assert "Arm3" in observed["reason_zh"]
    assert runtime.camera_coordination.next_request() is None


def test_arm3_camera_recovery_resumes_the_blocked_primary_inspection() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.faults.isolated_resources["ARM3"] = None
    runtime.submit_order("A", order_id="CAMERA_RECOVERED_PRIMARY")

    for _index in range(2_000):
        runtime.tick(0.02)
        if runtime.units["CAMERA_RECOVERED_PRIMARY_UNIT_01"].stage.value == "MATERIAL_INSPECTION":
            break
    else:
        raise AssertionError("order did not block at S2B")

    assert runtime.camera_coordination.next_request() is None
    runtime.faults.isolated_resources.pop("ARM3")
    snapshot = _run(runtime)

    assert snapshot["complete"]
    assert snapshot["camera_coordination"]["review_history"] == []


def test_reset_clears_camera_requests_and_capture_audit() -> None:
    runtime = DualLineRuntime(fast=True)
    runtime.submit_plan(_plan("CAMERA_RESET", "HIGH_RELIABILITY"))
    _run(runtime)
    assert runtime.snapshot()["camera_coordination"]["review_history"]

    runtime.reset()

    coordination = runtime.snapshot()["camera_coordination"]
    assert coordination["pending_reviews"] == []
    assert coordination["review_history"] == []
    assert coordination["active_plan"]["status"] == "STANDBY"


def test_camera_coordination_snapshot_exposes_one_arm3_camera_for_all_robot_inspections() -> None:
    runtime = DualLineRuntime(fast=True)

    snapshot = runtime.snapshot()["camera_coordination"]

    stations = {item["station_id"]: item for item in snapshot["stations"]}
    assert stations["S2B_MATERIAL_INSPECTION"]["primary_camera"] == "ARM3_CAMERA"
    assert stations["S2B_MATERIAL_INSPECTION"]["secondary_camera"] is None
    assert stations["S4_PRE_BRAZE_INSPECTION"]["primary_camera"] == "ARM3_CAMERA"
    assert stations["S4_PRE_BRAZE_INSPECTION"]["secondary_camera"] is None
    assert stations["S3B_ARM3_INSTALL"]["primary_camera"] == "ARM3_CAMERA"
    assert stations["S3B_ARM3_INSTALL"]["inspection_kinds"] == [
        "MATERIAL_INSPECTION",
        "PRE_BRAZE_INSPECTION",
    ]
    assert stations["POST_BRAZE_SCAN"]["primary_camera"] == "POST_CAMERA"
    assert stations["POST_BRAZE_SCAN"]["secondary_camera"] is None


def test_v2_control_surface_accepts_reliable_camera_routes() -> None:
    runtime = DualLineRuntime(fast=True)
    surface = V2ControlSurface(runtime)

    surface.process(
        {
            "type": "order_insert",
            "order_id": "CAMERA_ROUTE_UI",
            "preset": "A",
            "quantity": 1,
            "route_strategy": "HIGH_RELIABILITY",
        }
    )

    assert runtime.orders["CAMERA_ROUTE_UI"].route_strategy == "HIGH_RELIABILITY"


def test_dynamic_second_camera_order_cannot_stall_the_arm3_installation_branch() -> None:
    """A queued second close-up must not strand the first tray on Arm3's branch."""

    args = parse_args(["--headless", "--fast", "--max-sim-time", "120"])
    application = V2BrazingApplication(args)
    inserted = False
    try:
        application.runtime.submit_order(
            "A",
            order_id="CAMERA_FIRST",
            route_strategy="HIGH_RELIABILITY",
        )
        for _index in range(4_800):
            application.advance_frame()
            operation = application.runtime.operations.get("ARM3")
            if (
                not inserted
                and operation is not None
                and operation.unit_id == "CAMERA_FIRST_UNIT_01"
                and operation.kind == "INSTALL_FIN"
            ):
                application.runtime.submit_order(
                    "B",
                    order_id="CAMERA_SECOND",
                    route_strategy="HIGH_RELIABILITY",
                )
                inserted = True
            if inserted and application.runtime.complete:
                break
        else:
            raise AssertionError(
                "two Arm3-camera orders stalled after dynamic insertion: "
                f"{application.runtime.snapshot()['arm3_camera_plan']}"
            )

        assert inserted
        assert all(unit.fins_installed == unit.fin_count for unit in application.runtime.units.values())
        assert application.runtime.complete
    finally:
        application.close()
