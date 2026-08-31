from __future__ import annotations

import pytest

from brazing_sim.dual_line.runtime import DualLineRuntime
from brazing_sim.events import EventType
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.twin import DecisionEvent, DigitalTwinSnapshot


def test_snapshot_is_deeply_immutable_and_has_stable_fingerprint() -> None:
    source = {
        "sim_time": 12.5,
        "orders": [{"order_id": "A", "status": "RUNNING"}],
        "resources_v2": {"ARM1": {"status": "BUSY"}},
    }
    snapshot = DigitalTwinSnapshot.from_mapping(source, source_name="test", plan_version=3)

    assert snapshot.sim_time == pytest.approx(12.5)
    assert snapshot.plan_version == 3
    assert snapshot.ready_task_ids == ()
    assert snapshot.active_resources == ("ARM1",)
    assert snapshot.fingerprint == DigitalTwinSnapshot.from_mapping(
        source, source_name="other", plan_version=3
    ).fingerprint

    with pytest.raises(TypeError):
        snapshot.state["sim_time"] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.orders[0]["status"] = "COMPLETE"  # type: ignore[index]

    exported = snapshot.as_dict()
    exported["orders"][0]["status"] = "COMPLETE"
    assert snapshot.orders[0]["status"] == "RUNNING"


def test_decision_event_has_one_serialisable_contract() -> None:
    event = DecisionEvent(
        event_type=EventType.PLAN_PROPOSED,
        sim_time=4.0,
        source="TwinShield-RH",
        plan_version=8,
        trigger="TASK_SUCCEEDED",
        task_ids=("TASK-01",),
        payload={"candidate_count": 2, "total_cost": 14.5},
    )

    encoded = event.as_dict()
    assert encoded["event_type"] == "PLAN_PROPOSED"
    assert encoded["plan_version"] == 8
    assert encoded["task_ids"] == ["TASK-01"]
    assert encoded["payload"]["candidate_count"] == 2


def test_runtime_adapters_capture_without_changing_runtime_state() -> None:
    manufacturing = ManufacturingRuntime()
    manufacturing_snapshot = manufacturing.capture_digital_twin()
    assert manufacturing_snapshot.source_name == "ManufacturingRuntime"
    assert manufacturing_snapshot.schema_version == 2
    assert manufacturing.events.history == []

    v2 = DualLineRuntime(fast=True)
    v2_snapshot = v2.capture_digital_twin()
    assert v2_snapshot.source_name == "DualLineRuntime"
    assert v2_snapshot.sim_time == pytest.approx(0.0)
    assert v2.events == []


def test_runtime_adapter_can_emit_explicit_snapshot_event() -> None:
    runtime = ManufacturingRuntime()
    snapshot = runtime.capture_digital_twin(emit_event=True)

    assert snapshot.fingerprint
    assert len(runtime.events.history) == 1
    assert runtime.events.history[0].event_type is EventType.STATE_SNAPSHOT_CAPTURED
    assert runtime.events.history[0].payload["fingerprint"] == snapshot.fingerprint

    v2 = DualLineRuntime(fast=True)
    v2.capture_digital_twin(emit_event=True)
    assert v2.events[-1]["type"] == EventType.STATE_SNAPSHOT_CAPTURED.value
