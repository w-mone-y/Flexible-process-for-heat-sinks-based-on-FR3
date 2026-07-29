from __future__ import annotations

from dataclasses import dataclass, field

from brazing_sim.dual_line.runtime import DualLineRuntime
from brazing_sim.dual_line.tray_flow import TrayOwner


@dataclass
class _ControllableGate:
    ready_owners: set[tuple[str, TrayOwner]] = field(default_factory=set)
    completed_operations: set[tuple[str, str, str]] = field(default_factory=set)

    def tray_ready(self, tray_id: str, owner: TrayOwner) -> bool:
        return (tray_id, owner) in self.ready_owners

    def operation_complete(self, resource: str, unit_id: str, kind: str) -> bool:
        return (resource, unit_id, kind) in self.completed_operations

    def owner_available(self, owner: TrayOwner) -> bool:
        return True


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
