from __future__ import annotations

from dataclasses import dataclass, field

from brazing_sim.dual_line import (
    DualLineRuntime,
    InstallBranch,
    TrayPhase,
    UnitStage,
)
from brazing_sim.dual_line.tray_flow import TrayOwner
from brazing_sim.dual_line.unified_runtime import V2PhysicalExecutionBridge
from brazing_sim.planning import ManufacturingTask, TaskType


@dataclass
class _ControllableGate:
    ready_owners: set[tuple[str, TrayOwner]] = field(default_factory=set)
    completed_operations: set[tuple[str, str, str]] = field(default_factory=set)
    allowed_operations: set[tuple[str, str, str]] | None = None

    def tray_ready(self, tray_id: str, owner: TrayOwner) -> bool:
        return (tray_id, owner) in self.ready_owners

    def operation_complete(self, resource: str, unit_id: str, kind: str) -> bool:
        return (resource, unit_id, kind) in self.completed_operations

    def owner_available(self, owner: TrayOwner) -> bool:
        return True

    def operation_start_allowed(self, resource: str, unit_id: str, kind: str) -> bool:
        return (
            self.allowed_operations is None
            or (
                resource,
                unit_id,
                kind,
            )
            in self.allowed_operations
        )


def test_v2_runtime_never_starts_an_operation_without_external_authority() -> None:
    runtime = DualLineRuntime(fast=True)
    gate = _ControllableGate(allowed_operations=set())
    runtime.set_execution_gate(gate)

    order = runtime.submit_order("A", order_id="PHYSICAL_AUTHORITY")
    unit = runtime.units[order.unit_ids[0]]
    assert unit.tray_id is not None
    gate.ready_owners.add((unit.tray_id, TrayOwner.S1))
    runtime.tick(0.05)

    assert "ARM1" not in runtime.operations

    gate.allowed_operations.add(("ARM1", unit.unit_id, "BASE_LOADING"))
    runtime.tick(0.05)

    assert runtime.operations["ARM1"].kind == "BASE_LOADING"


def test_v2_runtime_does_not_start_station_work_before_the_tray_physically_arrives() -> None:
    runtime = DualLineRuntime(fast=True)
    gate = _ControllableGate()
    runtime.set_execution_gate(gate)

    order = runtime.submit_order("A", order_id="PHYSICAL_START")
    unit = runtime.units[order.unit_ids[0]]
    assert unit.tray_id is not None
    assert "ARM1" not in runtime.operations

    gate.ready_owners.add((unit.tray_id, TrayOwner.S1))
    runtime.tick(0.05)

    assert runtime.operations["ARM1"].kind == "BASE_LOADING"


def test_v2_runtime_keeps_an_expired_operation_running_until_the_actor_is_done() -> None:
    runtime = DualLineRuntime(fast=True)
    gate = _ControllableGate()
    runtime.set_execution_gate(gate)
    order = runtime.submit_order("A", order_id="PHYSICAL_COMPLETE")
    unit = runtime.units[order.unit_ids[0]]
    assert unit.tray_id is not None
    gate.ready_owners.add((unit.tray_id, TrayOwner.S1))
    runtime.tick(0.05)

    runtime.tick(runtime.durations.base_load + 0.1)
    assert runtime.operations["ARM1"].remaining_s == 0.0
    assert unit.stage.value == "BASE_LOADING"

    gate.completed_operations.add(("ARM1", unit.unit_id, "BASE_LOADING"))
    runtime.tick(0.05)

    assert "ARM1" not in runtime.operations
    assert any(
        event["type"] == "OPERATION_COMPLETED"
        and event["kind"] == "BASE_LOADING"
        and event["unit_id"] == unit.unit_id
        for event in runtime.events
    )


def test_s1_and_s2a_pipeline_while_arm2_prepositions_for_the_incoming_tray() -> None:
    runtime = DualLineRuntime(fast=True)
    gate = _ControllableGate()
    runtime.set_execution_gate(gate)
    first = runtime.submit_order("A", order_id="PIPELINE_FIRST")
    second = runtime.submit_order("B", order_id="PIPELINE_SECOND")
    first_unit = runtime.units[first.unit_ids[0]]
    second_unit = runtime.units[second.unit_ids[0]]
    assert first_unit.tray_id is not None
    gate.ready_owners.add((first_unit.tray_id, TrayOwner.S1))

    runtime.tick(0.05)
    gate.completed_operations.add(("ARM1", first_unit.unit_id, "BASE_LOADING"))
    runtime.tick(runtime.durations.base_load + 0.05)

    assert first_unit.stage is UnitStage.DISPENSING
    assert second_unit.stage is UnitStage.BASE_LOADING
    assert second_unit.tray_id is not None
    gate.ready_owners.add((second_unit.tray_id, TrayOwner.S1))
    runtime.tick(0.05)

    snapshot = runtime.snapshot()
    owners = {tray["unit_id"]: tray["owner"] for tray in snapshot["trays"] if tray["unit_id"]}
    assert owners[first_unit.unit_id] == TrayOwner.S2A.value
    assert owners[second_unit.unit_id] == TrayOwner.S1.value
    assert runtime.operations["ARM1"].unit_id == second_unit.unit_id
    assert "ARM2" not in runtime.operations
    assert snapshot["prepositioning"]["ARM2"] == {
        "unit_id": first_unit.unit_id,
        "operation_kind": "DISPENSING",
        "station_id": "S2A_DISPENSING",
        "reason_zh": "托盘运输至S2A时提前到达安全接近位",
    }
    assert snapshot["metrics"]["robot_transport_overlap_s"] > 0.0

    gate.ready_owners.add((first_unit.tray_id, TrayOwner.S2A))
    runtime.tick(0.05)

    assert runtime.operations["ARM2"].unit_id == first_unit.unit_id
    arrived = runtime.snapshot()
    assert "ARM2" not in arrived["prepositioning"]
    assert arrived["metrics"]["s1_s2a_dual_occupancy_s"] > 0.0


def test_arm1_prepares_gripper_immediately_after_the_last_base_is_loaded() -> None:
    runtime = DualLineRuntime(fast=True)
    gate = _ControllableGate()
    runtime.set_execution_gate(gate)
    first = runtime.submit_order("A", order_id="TAIL_FIN_FIRST")
    second = runtime.submit_order("B", order_id="TAIL_FIN_SECOND")
    first_unit = runtime.units[first.unit_ids[0]]
    second_unit = runtime.units[second.unit_ids[0]]
    assert first_unit.tray_id is not None

    gate.ready_owners.add((first_unit.tray_id, TrayOwner.S1))
    runtime.tick(0.05)
    gate.completed_operations.add(("ARM1", first_unit.unit_id, "BASE_LOADING"))
    runtime.tick(runtime.durations.base_load + 0.05)

    assert first_unit.stage is UnitStage.DISPENSING
    assert second_unit.stage is UnitStage.BASE_LOADING
    assert second_unit.tray_id is not None
    assert runtime.snapshot()["prepositioning"]["ARM1"]["operation_kind"] == "BASE_LOADING"

    gate.ready_owners.add((second_unit.tray_id, TrayOwner.S1))
    runtime.tick(0.05)
    gate.completed_operations.add(("ARM1", second_unit.unit_id, "BASE_LOADING"))
    runtime.tick(runtime.durations.base_load + 0.05)

    assert first_unit.stage is UnitStage.DISPENSING
    assert second_unit.stage is UnitStage.WAITING_S2A
    assert "ARM1" not in runtime.operations
    intent = runtime.snapshot()["prepositioning"]["ARM1"]
    assert intent["operation_kind"] == "INSTALL_FIN"
    assert intent["station_id"] == "S3A_ARM1_INSTALL"
    assert "最后一块基板" in intent["reason_zh"]


def test_preposition_intents_cover_s1_base_loading_and_s2b_camera_approach() -> None:
    runtime = DualLineRuntime(fast=True)
    gate = _ControllableGate()
    runtime.set_execution_gate(gate)
    order = runtime.submit_order("A", order_id="MULTI_STATION_PREPOSITION")
    unit = runtime.units[order.unit_ids[0]]
    assert unit.tray_id is not None

    base_intent = runtime.snapshot()["prepositioning"]["ARM1"]
    assert base_intent["operation_kind"] == "BASE_LOADING"
    assert "ARM1" not in runtime.operations

    gate.ready_owners.add((unit.tray_id, TrayOwner.S1))
    runtime.tick(0.05)
    gate.completed_operations.add(("ARM1", unit.unit_id, "BASE_LOADING"))
    runtime.tick(runtime.durations.base_load + 0.05)
    gate.ready_owners.add((unit.tray_id, TrayOwner.S2A))
    runtime.tick(0.05)
    gate.completed_operations.add(("ARM2", unit.unit_id, "DISPENSING"))
    runtime.tick(runtime.durations.dispensing + 0.05)

    camera_intent = runtime.snapshot()["prepositioning"]["ARM3"]
    assert camera_intent == {
        "unit_id": unit.unit_id,
        "operation_kind": "MATERIAL_INSPECTION",
        "station_id": "S2B_MATERIAL_INSPECTION",
        "reason_zh": "托盘运输至S2B时提前到达相机安全位",
    }
    assert "ARM3" not in runtime.operations


def test_selected_fin_branch_prepositions_its_robot_while_the_tray_moves_from_s2b() -> None:
    runtime = DualLineRuntime(fast=True)
    gate = _ControllableGate()
    runtime.set_execution_gate(gate)
    order = runtime.submit_order("A", order_id="FIN_BRANCH_PREPOSITION")
    unit = runtime.units[order.unit_ids[0]]
    assert unit.tray_id is not None

    gate.ready_owners.add((unit.tray_id, TrayOwner.S1))
    runtime.tick(0.05)
    gate.completed_operations.add(("ARM1", unit.unit_id, "BASE_LOADING"))
    runtime.tick(runtime.durations.base_load + 0.05)
    gate.ready_owners.add((unit.tray_id, TrayOwner.S2A))
    runtime.tick(0.05)
    gate.completed_operations.add(("ARM2", unit.unit_id, "DISPENSING"))
    runtime.tick(runtime.durations.dispensing + 0.05)
    gate.ready_owners.add((unit.tray_id, TrayOwner.S2B))
    runtime.tick(0.05)
    gate.completed_operations.add(("ARM3", unit.unit_id, "MATERIAL_INSPECTION"))
    runtime.tick(runtime.durations.material_inspection + 0.05)

    assert unit.stage is UnitStage.FIN_INSTALLATION
    assert unit.branch is not None
    selected_resource = "ARM1" if unit.branch is InstallBranch.ARM1_A else "ARM3"
    intent = runtime.snapshot()["prepositioning"][selected_resource]
    assert intent["unit_id"] == unit.unit_id
    assert intent["operation_kind"] == "INSTALL_FIN"
    assert intent["station_id"] == ("S3A_ARM1_INSTALL" if selected_resource == "ARM1" else "S3B_ARM3_INSTALL")
    assert selected_resource not in runtime.operations


def test_furnace_unload_permit_is_bound_to_the_exact_product_unit() -> None:
    bridge = V2PhysicalExecutionBridge(DualLineRuntime(fast=True))
    task = ManufacturingTask(
        task_id="EXACT_UNLOAD",
        task_type=TaskType.UNLOAD_RACK_LAYER,
        order_id="EXACT_PERMIT",
        unit_id="EXACT_PERMIT_UNIT_01",
        eligible_resources=["FURNACE_TRANSFER"],
    )
    bridge.authorize(task, "FURNACE_TRANSFER")

    assert bridge.operation_start_allowed(
        "FURNACE_TRANSFER",
        "EXACT_PERMIT_UNIT_01",
        "FURNACE_UNLOAD_TRAY",
    )
    assert not bridge.operation_start_allowed(
        "FURNACE_TRANSFER",
        "DIFFERENT_UNIT_01",
        "FURNACE_UNLOAD_TRAY",
    )


def test_completed_pick_fin_revokes_its_physical_start_permit() -> None:
    runtime = DualLineRuntime(fast=True)
    order = runtime.submit_order("A", order_id="SINGLE_FIN_PERMIT")
    unit_id = order.unit_ids[0]
    bridge = V2PhysicalExecutionBridge(runtime)
    bridge.bind_physical_gate(_ControllableGate())
    task = ManufacturingTask(
        task_id="SINGLE_FIN_PICK_01",
        task_type=TaskType.PICK_FIN,
        order_id=order.order_id,
        unit_id=unit_id,
        eligible_resources=["ARM3"],
        payload={"fin_id": "fin_01"},
    )
    skill = bridge.build_registry().create(TaskType.PICK_FIN)
    skill.start(task, "ARM3", bridge, now=0.0)

    assert bridge.operation_start_allowed("ARM3", unit_id, "INSTALL_FIN")

    runtime._event("FIN_INSTALLED", unit_id=unit_id, fin_index=1, fin_count=5)
    result = skill.update(now=0.1, dt=0.1)

    assert result.succeeded
    assert not bridge.operation_start_allowed("ARM3", unit_id, "INSTALL_FIN")


def test_appended_furnace_batch_member_completes_elevator_gate_after_physical_loading() -> None:
    runtime = DualLineRuntime(fast=True)
    leader_order = runtime.submit_order("A", order_id="BATCH_LEADER")
    appended_order = runtime.submit_order("B", order_id="BATCH_APPENDED")
    outsider_order = runtime.submit_order("C", order_id="NEXT_BATCH")
    leader_id = leader_order.unit_ids[0]
    appended_id = appended_order.unit_ids[0]
    outsider_id = outsider_order.unit_ids[0]
    runtime._active_batch_units = [leader_id, appended_id]
    runtime.units[appended_id].stage = UnitStage.BRAZING
    runtime.units[outsider_id].stage = UnitStage.BRAZING
    bridge = V2PhysicalExecutionBridge(runtime)

    appended_task = ManufacturingTask(
        task_id="APPENDED_MOVE_ELEVATOR",
        task_type=TaskType.MOVE_ELEVATOR,
        order_id="BATCH_APPENDED",
        unit_id=appended_id,
        eligible_resources=["FURNACE_TRANSFER"],
    )
    outsider_task = ManufacturingTask(
        task_id="OUTSIDER_MOVE_ELEVATOR",
        task_type=TaskType.MOVE_ELEVATOR,
        order_id="NEXT_BATCH",
        unit_id=outsider_id,
        eligible_resources=["FURNACE_TRANSFER"],
    )

    assert bridge.task_complete(appended_task, "FURNACE_TRANSFER", since=0.0)[0]
    assert not bridge.task_complete(outsider_task, "FURNACE_TRANSFER", since=0.0)[0]


def test_base_pick_does_not_reserve_arm1_before_its_tray_reaches_s1() -> None:
    runtime = DualLineRuntime(fast=True)
    order = runtime.submit_order("A", order_id="S1_PHYSICAL_READY")
    unit_id = order.unit_ids[0]
    bridge = V2PhysicalExecutionBridge(runtime)
    task = ManufacturingTask(
        task_id="S1_PICK_BASE",
        task_type=TaskType.PICK_BASE_PLATE,
        order_id="S1_PHYSICAL_READY",
        unit_id=unit_id,
        eligible_resources=["ARM1"],
    )
    runtime.units[unit_id].stage = UnitStage.QUEUED

    assert bridge.task_dispatch_allowed(task)[0] is False

    runtime.units[unit_id].stage = UnitStage.BASE_LOADING

    assert bridge.task_dispatch_allowed(task) == (True, "")


def test_denied_fin_start_does_not_prevent_arm1_from_staging_the_next_base() -> None:
    runtime = DualLineRuntime(fast=True)
    gate = _ControllableGate(allowed_operations=set())
    runtime.set_execution_gate(gate)
    fin_order = runtime.submit_order("A", order_id="FIN_NOT_SELECTED")
    base_order = runtime.submit_order("B", order_id="BASE_SHOULD_STAGE")
    fin_unit = runtime.units[fin_order.unit_ids[0]]
    base_unit = runtime.units[base_order.unit_ids[0]]
    assert fin_unit.tray_id is not None
    runtime.flow.handoff(
        fin_unit.tray_id,
        TrayOwner.S1,
        TrayOwner.S2A,
        TrayPhase.DISPENSING,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        fin_unit.tray_id,
        TrayOwner.S2A,
        TrayOwner.S2B,
        TrayPhase.MATERIAL_INSPECTION,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        fin_unit.tray_id,
        TrayOwner.S2B,
        TrayOwner.INSTALL_A,
        TrayPhase.FIN_INSTALLATION,
        now=runtime.sim_time,
    )
    fin_unit.stage = UnitStage.FIN_INSTALLATION
    fin_unit.branch = InstallBranch.ARM1_A
    gate.ready_owners.add((fin_unit.tray_id, TrayOwner.INSTALL_A))

    runtime.tick(0.05)

    assert "ARM1" not in runtime.operations
    assert base_unit.stage is UnitStage.BASE_LOADING
    assert base_unit.tray_id is not None
    assert runtime.flow.get(base_unit.tray_id).owner is TrayOwner.S1


def test_arm3_inspection_permit_is_bound_to_the_exact_product_unit() -> None:
    runtime = DualLineRuntime(fast=True)
    permitted_order = runtime.submit_order("A", order_id="INSPECTION_PERMITTED")
    other_order = runtime.submit_order("B", order_id="INSPECTION_OTHER")
    permitted_unit = permitted_order.unit_ids[0]
    other_unit = other_order.unit_ids[0]
    bridge = V2PhysicalExecutionBridge(runtime)
    task = ManufacturingTask(
        task_id="EXACT_MATERIAL_INSPECTION",
        task_type=TaskType.INSPECT_BRAZING,
        order_id="INSPECTION_PERMITTED",
        unit_id=permitted_unit,
        eligible_resources=["ARM3"],
    )
    bridge.authorize(task, "ARM3")

    assert bridge.operation_start_allowed(
        "ARM3",
        permitted_unit,
        "MATERIAL_INSPECTION",
    )
    assert not bridge.operation_start_allowed(
        "ARM3",
        other_unit,
        "MATERIAL_INSPECTION",
    )


def test_material_inspection_does_not_reserve_arm3_before_the_tray_reaches_s2b() -> None:
    runtime = DualLineRuntime(fast=True)
    order = runtime.submit_order("A", order_id="S2B_PHYSICAL_READY")
    unit_id = order.unit_ids[0]
    bridge = V2PhysicalExecutionBridge(runtime)
    task = ManufacturingTask(
        task_id="S2B_INSPECT_BRAZING",
        task_type=TaskType.INSPECT_BRAZING,
        order_id=order.order_id,
        unit_id=unit_id,
        eligible_resources=["ARM3"],
    )
    runtime.units[unit_id].stage = UnitStage.WAITING_S2B

    allowed, reason = bridge.task_dispatch_allowed(task)

    assert not allowed
    assert "S2B" in reason

    runtime.units[unit_id].stage = UnitStage.MATERIAL_INSPECTION

    assert bridge.task_dispatch_allowed(task) == (True, "")


def test_fin_pick_does_not_reserve_a_robot_before_the_tray_reaches_installation() -> None:
    runtime = DualLineRuntime(fast=True)
    order = runtime.submit_order("A", order_id="FIN_PHYSICAL_READY")
    unit_id = order.unit_ids[0]
    bridge = V2PhysicalExecutionBridge(runtime)
    task = ManufacturingTask(
        task_id="FIN_PHYSICAL_READY_UNIT_01_PICK_FIN_01",
        task_type=TaskType.PICK_FIN,
        order_id=order.order_id,
        unit_id=unit_id,
        eligible_resources=["ARM1", "ARM3"],
    )
    runtime.units[unit_id].stage = UnitStage.WAITING_INSTALL

    allowed, reason = bridge.task_dispatch_allowed(task)

    assert not allowed
    assert "装配位" in reason

    runtime.units[unit_id].stage = UnitStage.FIN_INSTALLATION

    assert bridge.task_dispatch_allowed(task) == (True, "")


def test_arm3_tries_the_globally_permitted_inspection_when_another_camera_candidate_is_denied() -> None:
    runtime = DualLineRuntime(fast=True)
    gate = _ControllableGate(allowed_operations=set())
    runtime.set_execution_gate(gate)
    s4_order = runtime.submit_order("A", order_id="S4_NOT_PERMITTED")
    s2b_order = runtime.submit_order("B", order_id="S2B_PERMITTED")
    s4_unit = runtime.units[s4_order.unit_ids[0]]
    s2b_unit = runtime.units[s2b_order.unit_ids[0]]
    assert s4_unit.tray_id is not None
    runtime.flow.handoff(
        s4_unit.tray_id,
        TrayOwner.S1,
        TrayOwner.S2A,
        TrayPhase.DISPENSING,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        s4_unit.tray_id,
        TrayOwner.S2A,
        TrayOwner.S2B,
        TrayPhase.MATERIAL_INSPECTION,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        s4_unit.tray_id,
        TrayOwner.S2B,
        TrayOwner.INSTALL_A,
        TrayPhase.FIN_INSTALLATION,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        s4_unit.tray_id,
        TrayOwner.INSTALL_A,
        TrayOwner.MERGE_A_WAIT,
        TrayPhase.MERGE_WAIT,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        s4_unit.tray_id,
        TrayOwner.MERGE_A_WAIT,
        TrayOwner.MERGE,
        TrayPhase.MERGING,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        s4_unit.tray_id,
        TrayOwner.MERGE,
        TrayOwner.S4,
        TrayPhase.PRE_BRAZE_INSPECTION,
        now=runtime.sim_time,
    )
    s4_unit.stage = UnitStage.PRE_BRAZE_INSPECTION
    s2b_tray = runtime.flow.assign_order(
        s2b_unit.order_id,
        s2b_unit.unit_id,
        now=runtime.sim_time,
    )
    s2b_unit.tray_id = s2b_tray.tray_id
    runtime.flow.handoff(
        s2b_tray.tray_id,
        TrayOwner.S1,
        TrayOwner.S2A,
        TrayPhase.DISPENSING,
        now=runtime.sim_time,
    )
    runtime.flow.handoff(
        s2b_tray.tray_id,
        TrayOwner.S2A,
        TrayOwner.S2B,
        TrayPhase.MATERIAL_INSPECTION,
        now=runtime.sim_time,
    )
    s2b_unit.stage = UnitStage.MATERIAL_INSPECTION
    gate.ready_owners.update(
        {
            (s4_unit.tray_id, TrayOwner.S4),
            (s2b_unit.tray_id, TrayOwner.S2B),
        }
    )
    gate.allowed_operations.add(("ARM3", s2b_unit.unit_id, "MATERIAL_INSPECTION"))

    runtime.tick(0.05)

    assert runtime.operations["ARM3"].unit_id == s2b_unit.unit_id
    assert runtime.operations["ARM3"].kind == "MATERIAL_INSPECTION"
