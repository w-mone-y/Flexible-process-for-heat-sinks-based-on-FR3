"""Tick-driven multi-order runtime for the independent V2 line.

This module is intentionally MuJoCo-independent.  It is the authoritative
logical state machine consumed by both the headless runner and the physical
scene adapter; rendering never gets to invent a tray transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Protocol

from .dispatch import (
    DualInstallDispatcher,
    InstallBranch,
    InstallRequest,
    InstallResourceState,
)
from .furnace import BatchRecipe, FurnacePhase, ThroughBatchFurnace
from .topology import DualLineTopology
from .tray_flow import TrayFlowController, TrayOwner, TrayPhase


class UnitStage(str, Enum):
    QUEUED = "QUEUED"
    BASE_LOADING = "BASE_LOADING"
    WAITING_S2A = "WAITING_S2A"
    DISPENSING = "DISPENSING"
    WAITING_S2B = "WAITING_S2B"
    MATERIAL_INSPECTION = "MATERIAL_INSPECTION"
    WAITING_INSTALL = "WAITING_INSTALL"
    FIN_INSTALLATION = "FIN_INSTALLATION"
    WAITING_MERGE = "WAITING_MERGE"
    MERGING = "MERGING"
    WAITING_S4 = "WAITING_S4"
    PRE_BRAZE_INSPECTION = "PRE_BRAZE_INSPECTION"
    WAITING_BUFFER = "WAITING_BUFFER"
    FURNACE_BUFFER = "FURNACE_BUFFER"
    FURNACE_LOADING = "FURNACE_LOADING"
    BRAZING = "BRAZING"
    FURNACE_UNLOADING = "FURNACE_UNLOADING"
    POST_BRAZE_INSPECTION = "POST_BRAZE_INSPECTION"
    WAITING_OUTPUT = "WAITING_OUTPUT"
    DELIVERING = "DELIVERING"
    PRODUCT_REMOVED = "PRODUCT_REMOVED"
    VIRTUAL_RETURN = "VIRTUAL_RETURN"
    COMPLETE = "COMPLETE"


@dataclass(slots=True)
class V2UnitState:
    unit_id: str
    order_id: str
    preset: str
    fin_count: int
    priority: int
    due_at: float | None
    urgent: bool = False
    stage: UnitStage = UnitStage.QUEUED
    tray_id: str | None = None
    branch: InstallBranch | None = None
    fins_installed: int = 0
    stage_started_at: float = 0.0
    completed_at: float | None = None
    buffer_owner: TrayOwner | None = None
    furnace_layer: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "order_id": self.order_id,
            "preset": self.preset,
            "fin_count": self.fin_count,
            "priority": self.priority,
            "due_at": self.due_at,
            "urgent": self.urgent,
            "stage": self.stage.value,
            "tray_id": self.tray_id,
            "branch": None if self.branch is None else self.branch.value,
            "fins_installed": self.fins_installed,
            "furnace_layer": self.furnace_layer,
            "completed_at": self.completed_at,
        }


@dataclass(slots=True)
class V2OrderState:
    order_id: str
    preset: str
    priority: int
    unit_ids: tuple[str, ...]
    inserted_at: float
    due_at: float | None = None
    urgent: bool = False

    def as_dict(self, units: dict[str, V2UnitState]) -> dict[str, object]:
        complete = all(units[unit_id].stage is UnitStage.COMPLETE for unit_id in self.unit_ids)
        return {
            "order_id": self.order_id,
            "preset": self.preset,
            "priority": self.priority,
            "due_at": self.due_at,
            "urgent": self.urgent,
            "unit_ids": list(self.unit_ids),
            "complete": complete,
        }


@dataclass(slots=True)
class _Operation:
    resource: str
    unit_id: str
    kind: str
    remaining_s: float
    started_at: float


class RuntimeExecutionGate(Protocol):
    """Narrow physical-feedback seam used by the MuJoCo application.

    The logical runtime remains usable without MuJoCo.  When a gate is bound,
    however, station work may only start after the pallet has physically
    reached its logical owner and an elapsed operation may only complete after
    its physical actor confirms completion.
    """

    def tray_ready(self, tray_id: str, owner: TrayOwner) -> bool: ...

    def operation_complete(self, resource: str, unit_id: str, kind: str) -> bool: ...

    def owner_available(self, owner: TrayOwner) -> bool: ...

    def operation_start_allowed(
        self,
        resource: str,
        unit_id: str,
        kind: str,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class _Durations:
    base_load: float
    dispensing: float
    material_inspection: float
    arm1_fin: float
    arm3_fin: float
    merge: float
    pre_braze_inspection: float
    post_braze_inspection: float
    output_delivery: float
    virtual_return: float
    furnace_door: float
    furnace_transfer: float

    @classmethod
    def for_mode(cls, fast: bool) -> "_Durations":
        scale = 0.35 if fast else 1.0
        return cls(
            base_load=1.4 * scale,
            dispensing=2.4 * scale,
            material_inspection=0.9 * scale,
            arm1_fin=1.3 * scale,
            arm3_fin=1.1 * scale,
            merge=1.0 * scale,
            pre_braze_inspection=1.2 * scale,
            post_braze_inspection=1.0 * scale,
            output_delivery=0.9 * scale,
            virtual_return=0.5 * scale,
            furnace_door=0.4 if fast else 1.0,
            furnace_transfer=0.6 if fast else 2.0,
        )


_FIN_COUNTS = {"A": 5, "B": 4, "C": 7}
_STANDARD_RECIPE = BatchRecipe("CAB_STANDARD", "aluminium", 600.0, 240.0, 0.10)


class DualLineRuntime:
    """Asynchronous six-tray runtime with shared Arm3 and a batch furnace."""

    def __init__(self, *, fast: bool = False) -> None:
        self.fast = bool(fast)
        self.topology = DualLineTopology.standard()
        topology_errors = self.topology.validate()
        if topology_errors:
            raise ValueError("; ".join(topology_errors))
        self.dispatcher = DualInstallDispatcher()
        self.flow = TrayFlowController(capacity=6)
        self.furnace = ThroughBatchFurnace(
            capacity=3,
            demo_cycle_seconds=30.0,
            real_cycle_seconds=3600.0,
            # ``fast`` shortens robot rehearsal durations; it must not change
            # the batching policy or split a three-unit order into premature
            # one-tray cycles merely because physical actors take longer.
            nominal_max_wait_seconds=600.0,
        )
        self.durations = _Durations.for_mode(fast)
        self.orders: dict[str, V2OrderState] = {}
        self.units: dict[str, V2UnitState] = {}
        self.operations: dict[str, _Operation] = {}
        self.events: list[dict[str, Any]] = []
        self.sim_time = 0.0
        self.paused = False
        self._order_sequence = 0
        self._active_batch_units: list[str] = []
        self._batch_sequence = 0
        self._batch_recorded = False
        self._furnace_load_queue: list[str] = []
        self._furnace_load_position = 0
        self._loading_batch_started_at = 0.0
        self._rear_door_ready = False
        self.completed_batches: list[dict[str, Any]] = []
        self.install_branch_counts = {branch: 0 for branch in InstallBranch}
        self.scheduled_parallel_install_seconds = 0.0
        self.upstream_work_during_brazing_s = 0.0
        self.maximum_wip = 0
        self._execution_gate: RuntimeExecutionGate | None = None

    def set_execution_gate(self, gate: RuntimeExecutionGate | None) -> None:
        """Bind physical readiness feedback, or restore logical-only mode."""

        self._execution_gate = gate

    @property
    def complete(self) -> bool:
        return bool(self.units) and all(unit.stage is UnitStage.COMPLETE for unit in self.units.values())

    def submit_order(
        self,
        preset: str,
        *,
        order_id: str | None = None,
        quantity: int = 1,
        priority: int = 10,
        due_at: float | None = None,
        urgent: bool = False,
    ) -> V2OrderState:
        preset = str(preset).strip().upper()
        if preset not in _FIN_COUNTS:
            raise ValueError("V2 preset must be A, B or C")
        if not 1 <= int(quantity) <= 3:
            raise ValueError("one V2 order may contain one to three units")
        if priority < 0:
            raise ValueError("priority must be non-negative")
        if due_at is not None and not isfinite(float(due_at)):
            raise ValueError("due time must be finite")
        self._order_sequence += 1
        identifier = order_id or f"V2_ORDER_{self._order_sequence:03d}"
        if identifier in self.orders:
            raise ValueError(f"duplicate V2 order id: {identifier}")
        unit_ids = tuple(f"{identifier}_UNIT_{index:02d}" for index in range(1, int(quantity) + 1))
        for unit_id in unit_ids:
            self.units[unit_id] = V2UnitState(
                unit_id=unit_id,
                order_id=identifier,
                preset=preset,
                fin_count=_FIN_COUNTS[preset],
                priority=int(priority),
                due_at=None if due_at is None else float(due_at),
                urgent=bool(urgent),
            )
        order = V2OrderState(
            order_id=identifier,
            preset=preset,
            priority=int(priority),
            unit_ids=unit_ids,
            inserted_at=self.sim_time,
            due_at=None if due_at is None else float(due_at),
            urgent=bool(urgent),
        )
        self.orders[identifier] = order
        self._event(
            "ORDER_QUEUED",
            order_id=identifier,
            preset=preset,
            quantity=quantity,
            priority=int(priority),
            due_at=order.due_at,
            urgent=order.urgent,
        )
        self._dispatch()
        return order

    def _event(self, event_type: str, **payload: Any) -> None:
        self.events.append({"time": round(self.sim_time, 6), "type": event_type, **payload})

    def _set_stage(self, unit: V2UnitState, stage: UnitStage) -> None:
        unit.stage = stage
        unit.stage_started_at = self.sim_time
        self._event(
            "UNIT_STAGE",
            unit_id=unit.unit_id,
            tray_id=unit.tray_id,
            stage=stage.value,
        )

    def _start(self, resource: str, unit: V2UnitState, kind: str, duration: float) -> None:
        if resource in self.operations:
            raise RuntimeError(f"resource already busy: {resource}")
        self.operations[resource] = _Operation(
            resource=resource,
            unit_id=unit.unit_id,
            kind=kind,
            remaining_s=float(duration),
            started_at=self.sim_time,
        )
        self._event("OPERATION_STARTED", resource=resource, unit_id=unit.unit_id, kind=kind)

    def _resource_available_at(self, resource: str) -> float:
        operation = self.operations.get(resource)
        return self.sim_time if operation is None else self.sim_time + operation.remaining_s

    def _waiting_units(self, *stages: UnitStage) -> list[V2UnitState]:
        allowed = set(stages)
        return sorted(
            (unit for unit in self.units.values() if unit.stage in allowed),
            key=lambda unit: (
                -int(unit.urgent),
                -unit.priority,
                float("inf") if unit.due_at is None else unit.due_at,
                unit.stage_started_at,
                unit.unit_id,
            ),
        )

    def _owner_free(self, owner: TrayOwner) -> bool:
        return not any(tray.owner is owner for tray in self.flow.trays)

    def _tray_ready(self, unit: V2UnitState) -> bool:
        if self._execution_gate is None:
            return True
        if unit.tray_id is None:
            return False
        tray = self.flow.get(unit.tray_id)
        return bool(self._execution_gate.tray_ready(unit.tray_id, tray.owner))

    def _target_available(self, owner: TrayOwner) -> bool:
        if self._execution_gate is None:
            return True
        return bool(self._execution_gate.owner_available(owner))

    def _operation_can_complete(self, operation: _Operation) -> bool:
        if self._execution_gate is None:
            return True
        return bool(
            self._execution_gate.operation_complete(
                operation.resource,
                operation.unit_id,
                operation.kind,
            )
        )

    def _operation_start_allowed(
        self,
        resource: str,
        unit: V2UnitState,
        kind: str,
    ) -> bool:
        if self._execution_gate is None:
            return True
        callback = getattr(self._execution_gate, "operation_start_allowed", None)
        if callback is None:
            return True
        return bool(callback(resource, unit.unit_id, kind))

    def _handoff(
        self,
        unit: V2UnitState,
        expected: TrayOwner,
        target: TrayOwner,
        phase: TrayPhase,
    ) -> None:
        assert unit.tray_id is not None
        self.flow.handoff(unit.tray_id, expected, target, phase, now=self.sim_time)
        self._event(
            "TRAY_HANDOFF",
            unit_id=unit.unit_id,
            tray_id=unit.tray_id,
            source=expected.value,
            target=target.value,
        )

    def _queued_fins(self, branch: InstallBranch, *, excluding: str) -> int:
        return sum(
            max(0, unit.fin_count - unit.fins_installed)
            for unit in self.units.values()
            if unit.unit_id != excluding
            and unit.branch is branch
            and unit.stage
            in {
                UnitStage.WAITING_INSTALL,
                UnitStage.FIN_INSTALLATION,
                UnitStage.WAITING_MERGE,
            }
        )

    def _assign_branch(self, unit: V2UnitState) -> InstallBranch:
        request = InstallRequest(
            tray_id=unit.tray_id or unit.unit_id,
            fin_count=unit.fin_count,
            ready_at=self.sim_time,
            due_at=unit.due_at,
            priority=unit.priority,
        )
        arm3_reservations: tuple[tuple[float, float], ...] = ()
        if self._waiting_units(UnitStage.WAITING_S2B, UnitStage.WAITING_S4):
            arm3_reservations = (
                (
                    self._resource_available_at("ARM3"),
                    self._resource_available_at("ARM3")
                    + self.durations.material_inspection
                    + self.durations.pre_braze_inspection,
                ),
            )
        decision = self.dispatcher.assign(
            request,
            (
                InstallResourceState(
                    InstallBranch.ARM1_A,
                    self._resource_available_at("ARM1"),
                    self.durations.arm1_fin,
                    queued_fins=self._queued_fins(InstallBranch.ARM1_A, excluding=unit.unit_id),
                ),
                InstallResourceState(
                    InstallBranch.ARM3_B,
                    self._resource_available_at("ARM3"),
                    self.durations.arm3_fin,
                    queued_fins=self._queued_fins(InstallBranch.ARM3_B, excluding=unit.unit_id),
                    inspection_reservations=arm3_reservations,
                ),
            ),
        )
        unit.branch = decision.branch
        self.install_branch_counts[decision.branch] += 1
        self._event(
            "INSTALL_ASSIGNED",
            unit_id=unit.unit_id,
            tray_id=unit.tray_id,
            branch=decision.branch.value,
            explanation_zh=decision.explanation_zh,
            selected_cost=decision.candidates[decision.branch].cost,
            candidates=[
                {
                    "resource_id": candidate.branch.value,
                    "start_at": candidate.start_at,
                    "finish_at": candidate.finish_at,
                    "queue_wait_s": candidate.queue_wait_s,
                    "inspection_wait_s": candidate.inspection_wait_s,
                    "lateness_s": candidate.lateness_s,
                    "cost": candidate.cost,
                    "reason": candidate.blocked_reason_zh,
                }
                for candidate in decision.candidates.values()
            ],
        )
        return decision.branch

    def _reroute_blocked_install(self, unit: V2UnitState) -> bool:
        """Move an S2B pallet to the other physically clear install branch.

        Dispatch cost is evaluated when material inspection finishes, but a
        previously completed pallet may reserve that branch's S3→S4 planar
        bypass before the inspected pallet starts moving.  Reassigning
        at S2B avoids a corridor deadlock while preserving branch immutability
        after the first physical handoff or fin installation.
        """

        if (
            unit.branch is None
            or unit.fins_installed != 0
            or unit.tray_id is None
            or self.flow.get(unit.tray_id).owner is not TrayOwner.S2B
        ):
            return False

        branch_owners = {
            InstallBranch.ARM1_A: TrayOwner.INSTALL_A,
            InstallBranch.ARM3_B: TrayOwner.INSTALL_B,
        }
        current_owner = branch_owners[unit.branch]
        if self._owner_free(current_owner) and self._target_available(current_owner):
            return False

        replacement = InstallBranch.ARM3_B if unit.branch is InstallBranch.ARM1_A else InstallBranch.ARM1_A
        replacement_owner = branch_owners[replacement]
        if not (self._owner_free(replacement_owner) and self._target_available(replacement_owner)):
            return False

        previous = unit.branch
        unit.branch = replacement
        self.install_branch_counts[previous] = max(
            0,
            self.install_branch_counts[previous] - 1,
        )
        self.install_branch_counts[unit.branch] += 1
        self._event(
            "INSTALL_REASSIGNED",
            unit_id=unit.unit_id,
            tray_id=unit.tray_id,
            previous_branch=previous.value,
            branch=unit.branch.value,
            reason_zh="原安装支路的平面绕障走廊被占用，改派另一空闲支路以释放S2B",
        )
        return True

    def _complete_operation(self, operation: _Operation) -> None:
        unit = self.units[operation.unit_id]
        self._event(
            "OPERATION_COMPLETED",
            resource=operation.resource,
            unit_id=unit.unit_id,
            kind=operation.kind,
        )
        if operation.kind == "BASE_LOADING":
            self._set_stage(unit, UnitStage.WAITING_S2A)
        elif operation.kind == "DISPENSING":
            self._set_stage(unit, UnitStage.WAITING_S2B)
        elif operation.kind == "MATERIAL_INSPECTION":
            self._set_stage(unit, UnitStage.WAITING_INSTALL)
            self._assign_branch(unit)
        elif operation.kind == "INSTALL_FIN":
            unit.fins_installed += 1
            self._event(
                "FIN_INSTALLED",
                unit_id=unit.unit_id,
                fin_index=unit.fins_installed,
                fin_count=unit.fin_count,
                branch=None if unit.branch is None else unit.branch.value,
            )
            if unit.fins_installed >= unit.fin_count:
                self._set_stage(unit, UnitStage.WAITING_MERGE)
            else:
                self._set_stage(unit, UnitStage.FIN_INSTALLATION)
        elif operation.kind == "MERGING":
            self._set_stage(unit, UnitStage.WAITING_S4)
        elif operation.kind == "PRE_BRAZE_INSPECTION":
            self._set_stage(unit, UnitStage.WAITING_BUFFER)
        elif operation.kind == "POST_BRAZE_INSPECTION":
            self._set_stage(unit, UnitStage.WAITING_OUTPUT)
        elif operation.kind == "OUTPUT_DELIVERY":
            assert unit.tray_id is not None
            self.flow.mark_product_removed(unit.tray_id, now=self.sim_time)
            self._set_stage(unit, UnitStage.PRODUCT_REMOVED)
        elif operation.kind == "VIRTUAL_RETURN":
            assert unit.tray_id is not None
            self.flow.complete_virtual_return(unit.tray_id, now=self.sim_time)
            self._set_stage(unit, UnitStage.COMPLETE)
            unit.completed_at = self.sim_time
            self._event("UNIT_COMPLETED", unit_id=unit.unit_id, order_id=unit.order_id)
        elif operation.kind == "FURNACE_FRONT_OPEN":
            self._event("FURNACE_FRONT_DOOR_OPENED")
        elif operation.kind == "FURNACE_LOAD_TRAY":
            assert unit.furnace_layer is not None
            self.furnace.lock_layer(unit.furnace_layer, now=self.sim_time)
            self._set_stage(unit, UnitStage.BRAZING)
            self._furnace_load_position += 1
        elif operation.kind == "FURNACE_FRONT_CLOSE":
            self.furnace.start_cycle(now=self.sim_time)
            self._event("FURNACE_THERMAL_CYCLE_STARTED")
        elif operation.kind == "FURNACE_REAR_OPEN":
            self._rear_door_ready = True
            self._event("FURNACE_REAR_DOOR_OPENED")
        elif operation.kind == "FURNACE_UNLOAD_TRAY":
            self._set_stage(unit, UnitStage.POST_BRAZE_INSPECTION)
            self._start(
                "POST_CAMERA",
                unit,
                "POST_BRAZE_INSPECTION",
                self.durations.post_braze_inspection,
            )
            if self.furnace.state.complete and not self._batch_recorded:
                record = {
                    "batch_id": f"V2_BATCH_{self._batch_sequence:03d}",
                    "unit_ids": list(self._active_batch_units),
                    "demo_cycle_s": self.furnace.demo_cycle_seconds,
                    "real_equivalent_cycle_s": self.furnace.real_cycle_seconds,
                }
                self.completed_batches.append(record)
                self._batch_recorded = True
                self._active_batch_units = []
                self._furnace_load_queue = []
                self._event("FURNACE_BATCH_COMPLETED", **record)

    def _advance_operations(self, dt: float) -> None:
        arm1_installing = (
            operation := self.operations.get("ARM1")
        ) is not None and operation.kind == "INSTALL_FIN"
        arm3_installing = (
            operation := self.operations.get("ARM3")
        ) is not None and operation.kind == "INSTALL_FIN"
        if arm1_installing and arm3_installing:
            self.scheduled_parallel_install_seconds += dt
        if self.furnace.state.phase in {
            FurnacePhase.PREHEAT,
            FurnacePhase.RAMP,
            FurnacePhase.SOAK,
            FurnacePhase.COOLING,
        } and any(operation.kind != "FURNACE" for operation in self.operations.values()):
            self.upstream_work_during_brazing_s += dt
        completed: list[_Operation] = []
        for operation in self.operations.values():
            operation.remaining_s = max(0.0, operation.remaining_s - dt)
            if operation.remaining_s <= 1.0e-9 and self._operation_can_complete(operation):
                completed.append(operation)
        for operation in completed:
            self.operations.pop(operation.resource, None)
            self._complete_operation(operation)

    def _admit_next_unit(self) -> bool:
        if (
            "ARM1" in self.operations
            or not self._owner_free(TrayOwner.S1)
            or not self._target_available(TrayOwner.S1)
        ):
            return False
        queued = self._waiting_units(UnitStage.QUEUED)
        if not queued:
            return False
        unit = queued[0]
        tray = self.flow.assign_order(unit.order_id, unit.unit_id, now=self.sim_time)
        unit.tray_id = tray.tray_id
        self._set_stage(unit, UnitStage.BASE_LOADING)
        return True

    def _dispatch_transfers(self) -> bool:
        changed = False
        for unit in self._waiting_units(UnitStage.WAITING_S2A):
            if (
                self._tray_ready(unit)
                and self._owner_free(TrayOwner.S2A)
                and self._target_available(TrayOwner.S2A)
            ):
                self._handoff(unit, TrayOwner.S1, TrayOwner.S2A, TrayPhase.DISPENSING)
                self._set_stage(unit, UnitStage.DISPENSING)
                changed = True
                break
        for unit in self._waiting_units(UnitStage.WAITING_S2B):
            if (
                self._tray_ready(unit)
                and self._owner_free(TrayOwner.S2B)
                and self._target_available(TrayOwner.S2B)
            ):
                self._handoff(
                    unit,
                    TrayOwner.S2A,
                    TrayOwner.S2B,
                    TrayPhase.MATERIAL_INSPECTION,
                )
                self._set_stage(unit, UnitStage.MATERIAL_INSPECTION)
                changed = True
                break
        for unit in self._waiting_units(UnitStage.WAITING_INSTALL):
            self._reroute_blocked_install(unit)
            if (
                self._tray_ready(unit)
                and unit.branch is InstallBranch.ARM1_A
                and self._owner_free(TrayOwner.INSTALL_A)
                and self._target_available(TrayOwner.INSTALL_A)
            ):
                self._handoff(
                    unit,
                    TrayOwner.S2B,
                    TrayOwner.INSTALL_A,
                    TrayPhase.FIN_INSTALLATION,
                )
                self._set_stage(unit, UnitStage.FIN_INSTALLATION)
                changed = True
                break
            if (
                self._tray_ready(unit)
                and unit.branch is InstallBranch.ARM3_B
                and self._owner_free(TrayOwner.INSTALL_B)
                and self._target_available(TrayOwner.INSTALL_B)
            ):
                self._handoff(
                    unit,
                    TrayOwner.S2B,
                    TrayOwner.INSTALL_B,
                    TrayPhase.FIN_INSTALLATION,
                )
                self._set_stage(unit, UnitStage.FIN_INSTALLATION)
                changed = True
                break
        for unit in self._waiting_units(UnitStage.WAITING_MERGE):
            assert unit.tray_id is not None
            if not self._tray_ready(unit):
                continue
            owner = self.flow.get(unit.tray_id).owner
            wait_owner = (
                TrayOwner.MERGE_A_WAIT if unit.branch is InstallBranch.ARM1_A else TrayOwner.MERGE_B_WAIT
            )
            source = TrayOwner.INSTALL_A if unit.branch is InstallBranch.ARM1_A else TrayOwner.INSTALL_B
            if owner is source and self._owner_free(wait_owner) and self._target_available(wait_owner):
                self._handoff(unit, source, wait_owner, TrayPhase.MERGE_WAIT)
                changed = True
                break
            if (
                owner is wait_owner
                and self._owner_free(TrayOwner.MERGE)
                and self._target_available(TrayOwner.MERGE)
            ):
                self._handoff(unit, wait_owner, TrayOwner.MERGE, TrayPhase.MERGING)
                self._set_stage(unit, UnitStage.MERGING)
                changed = True
                break
        for unit in self._waiting_units(UnitStage.WAITING_S4):
            if (
                self._tray_ready(unit)
                and self._owner_free(TrayOwner.S4)
                and self._target_available(TrayOwner.S4)
            ):
                self._handoff(
                    unit,
                    TrayOwner.MERGE,
                    TrayOwner.S4,
                    TrayPhase.PRE_BRAZE_INSPECTION,
                )
                self._set_stage(unit, UnitStage.PRE_BRAZE_INSPECTION)
                changed = True
                break
        for unit in self._waiting_units(UnitStage.WAITING_BUFFER):
            if not self._tray_ready(unit):
                continue
            # The three furnace buffers share one straight conveyor. Fill from
            # the far end so an arriving pallet never has to pass through an
            # occupied nearer position.
            for owner in (TrayOwner.BUFFER_3, TrayOwner.BUFFER_2, TrayOwner.BUFFER_1):
                if self._owner_free(owner) and self._target_available(owner):
                    self._handoff(unit, TrayOwner.S4, owner, TrayPhase.FURNACE_BUFFER)
                    unit.buffer_owner = owner
                    self._set_stage(unit, UnitStage.FURNACE_BUFFER)
                    changed = True
                    break
            if changed:
                break
        for unit in self._waiting_units(UnitStage.WAITING_OUTPUT):
            if (
                self._tray_ready(unit)
                and self._owner_free(TrayOwner.OUTPUT)
                and self._target_available(TrayOwner.OUTPUT)
            ):
                self._handoff(
                    unit,
                    TrayOwner.POST_SCAN,
                    TrayOwner.OUTPUT,
                    TrayPhase.DELIVERED,
                )
                self._set_stage(unit, UnitStage.DELIVERING)
                changed = True
                break
        return changed

    def _dispatch_resources(self) -> bool:
        changed = False
        if "ARM2" not in self.operations:
            waiting = [unit for unit in self._waiting_units(UnitStage.DISPENSING) if self._tray_ready(unit)]
            if waiting:
                self._start("ARM2", waiting[0], "DISPENSING", self.durations.dispensing)
                changed = True
        # Detection always gets Arm3 before another fin is picked. An active
        # INSTALL_FIN operation remains non-preemptible until this tick ends.
        if "ARM3" not in self.operations:
            waiting_s4 = [
                unit for unit in self._waiting_units(UnitStage.PRE_BRAZE_INSPECTION) if self._tray_ready(unit)
            ]
            if waiting_s4:
                self._start(
                    "ARM3",
                    waiting_s4[0],
                    "PRE_BRAZE_INSPECTION",
                    self.durations.pre_braze_inspection,
                )
                changed = True
            else:
                waiting_s2b = [
                    unit
                    for unit in self._waiting_units(UnitStage.MATERIAL_INSPECTION)
                    if self._tray_ready(unit)
                ]
                if waiting_s2b:
                    self._start(
                        "ARM3",
                        waiting_s2b[0],
                        "MATERIAL_INSPECTION",
                        self.durations.material_inspection,
                    )
                    changed = True
        if "MERGE" not in self.operations:
            waiting = [unit for unit in self._waiting_units(UnitStage.MERGING) if self._tray_ready(unit)]
            if waiting:
                self._start("MERGE", waiting[0], "MERGING", self.durations.merge)
                changed = True
        # Arm1 prioritises a tray already committed to installation; only then
        # does it admit another base plate.
        if "ARM1" not in self.operations:
            install = [
                unit
                for unit in self._waiting_units(UnitStage.FIN_INSTALLATION)
                if unit.branch is InstallBranch.ARM1_A
                and unit.fins_installed < unit.fin_count
                and self._tray_ready(unit)
            ]
            if install:
                self._start("ARM1", install[0], "INSTALL_FIN", self.durations.arm1_fin)
                changed = True
            else:
                base_loading = [
                    unit for unit in self._waiting_units(UnitStage.BASE_LOADING) if self._tray_ready(unit)
                ]
                if base_loading:
                    self._start(
                        "ARM1",
                        base_loading[0],
                        "BASE_LOADING",
                        self.durations.base_load,
                    )
                    changed = True
                elif self._admit_next_unit():
                    changed = True
        if "ARM3" not in self.operations:
            install = [
                unit
                for unit in self._waiting_units(UnitStage.FIN_INSTALLATION)
                if unit.branch is InstallBranch.ARM3_B
                and unit.fins_installed < unit.fin_count
                and self._tray_ready(unit)
            ]
            if install:
                self._start("ARM3", install[0], "INSTALL_FIN", self.durations.arm3_fin)
                changed = True
        if "OUTPUT" not in self.operations:
            delivering = [
                unit for unit in self._waiting_units(UnitStage.DELIVERING) if self._tray_ready(unit)
            ]
            if delivering:
                self._start(
                    "OUTPUT",
                    delivering[0],
                    "OUTPUT_DELIVERY",
                    self.durations.output_delivery,
                )
                changed = True
        for unit in self._waiting_units(UnitStage.VIRTUAL_RETURN):
            resource = f"RETURN:{unit.unit_id}"
            if resource not in self.operations:
                self._start(
                    resource,
                    unit,
                    "VIRTUAL_RETURN",
                    self.durations.virtual_return,
                )
                changed = True
        for unit in self._waiting_units(UnitStage.PRODUCT_REMOVED):
            assert unit.tray_id is not None
            self.flow.start_virtual_return(unit.tray_id, now=self.sim_time)
            self._set_stage(unit, UnitStage.VIRTUAL_RETURN)
            changed = True
        return changed

    def _units_outside_furnace_queue(self) -> list[V2UnitState]:
        settled = {
            UnitStage.FURNACE_BUFFER,
            UnitStage.FURNACE_LOADING,
            UnitStage.BRAZING,
            UnitStage.FURNACE_UNLOADING,
            UnitStage.POST_BRAZE_INSPECTION,
            UnitStage.WAITING_OUTPUT,
            UnitStage.DELIVERING,
            UnitStage.PRODUCT_REMOVED,
            UnitStage.VIRTUAL_RETURN,
            UnitStage.COMPLETE,
        }
        return [unit for unit in self.units.values() if unit.stage not in settled]

    def _try_start_furnace(self) -> bool:
        buffered = sorted(
            (
                unit
                for unit in self._waiting_units(UnitStage.FURNACE_BUFFER)
                if unit.unit_id not in self._active_batch_units
            ),
            key=lambda unit: (
                -int(unit.urgent),
                float("inf") if unit.due_at is None else unit.due_at,
                unit.stage_started_at,
                -unit.priority,
                unit.unit_id,
            ),
        )
        if not buffered:
            return False

        if self._active_batch_units:
            if (
                self.furnace.state.phase is not FurnacePhase.LOADING
                or len(self._active_batch_units) >= self.furnace.capacity
            ):
                return False
            unit = buffered[0]
            assert unit.tray_id is not None
            self.furnace.append_loading_tray(
                unit.tray_id,
                _STANDARD_RECIPE,
                now=self.sim_time,
            )
            self._active_batch_units.append(unit.unit_id)
            self._furnace_load_queue.append(unit.unit_id)
            self._event(
                "FURNACE_BATCH_TRAY_ADDED",
                batch_id=f"V2_BATCH_{self._batch_sequence:03d}",
                unit_id=unit.unit_id,
                tray_id=unit.tray_id,
                planned_count=len(self._active_batch_units),
            )
            return True

        if self.furnace.state.phase not in {FurnacePhase.IDLE, FurnacePhase.COMPLETE}:
            return False
        selected = buffered[:1]
        self.furnace.plan_batch(
            tuple((unit.tray_id or unit.unit_id, _STANDARD_RECIPE) for unit in selected),
            now=self.sim_time,
        )
        self.furnace.open_front(now=self.sim_time)
        self._active_batch_units = [unit.unit_id for unit in selected]
        buffer_rank = {
            TrayOwner.BUFFER_1: 1,
            TrayOwner.BUFFER_2: 2,
            TrayOwner.BUFFER_3: 3,
        }
        # The buffers are arranged in-line before the front door.  Loading the
        # nearest pallet first clears the corridor for the next two and avoids
        # driving a pallet through an occupied downstream buffer.
        self._furnace_load_queue = [
            unit.unit_id
            for unit in sorted(
                selected,
                key=lambda item: (
                    -buffer_rank.get(item.buffer_owner, 0),
                    item.unit_id,
                ),
            )
        ]
        self._furnace_load_position = 0
        self._loading_batch_started_at = self.sim_time
        self._rear_door_ready = False
        self._batch_sequence += 1
        self._batch_recorded = False
        self._start(
            "FURNACE_DOOR",
            selected[0],
            "FURNACE_FRONT_OPEN",
            self.durations.furnace_door,
        )
        self._event(
            "FURNACE_BATCH_STARTED",
            batch_id=f"V2_BATCH_{self._batch_sequence:03d}",
            unit_ids=list(self._active_batch_units),
        )
        return True

    def _loading_batch_ready_to_close(self) -> bool:
        if not self._active_batch_units:
            return False
        if len(self._active_batch_units) >= self.furnace.capacity:
            return True
        if not self._units_outside_furnace_queue():
            return True
        active = [self.units[unit_id] for unit_id in self._active_batch_units]
        due_values = [unit.due_at for unit in active if unit.due_at is not None]
        earliest_due_at = None if not due_values else min(due_values)
        estimated_remaining_s = (
            self.furnace.demo_cycle_seconds
            + 2.0 * self.durations.furnace_door
            + 2.0 * len(active) * self.durations.furnace_transfer
        )
        if earliest_due_at is not None and self.sim_time + estimated_remaining_s >= earliest_due_at:
            return True
        return self.sim_time - self._loading_batch_started_at >= self.furnace.nominal_max_wait_seconds

    def _dispatch_furnace_loading(self) -> bool:
        if (
            not self._active_batch_units
            or self.furnace.state.phase is not FurnacePhase.LOADING
            or "FURNACE_DOOR" in self.operations
            or "FURNACE_TRANSFER" in self.operations
        ):
            return False
        if self._furnace_load_position < len(self._furnace_load_queue):
            unit = self.units[self._furnace_load_queue[self._furnace_load_position]]
            assert unit.tray_id is not None and unit.buffer_owner is not None
            if not self._tray_ready(unit):
                return False
            layer = self.furnace.capacity - 1 - self._furnace_load_position
            unit.furnace_layer = layer
            self._handoff(
                unit,
                unit.buffer_owner,
                TrayOwner.FURNACE,
                TrayPhase.BRAZING,
            )
            self.furnace.load_front(unit.tray_id, layer=layer, now=self.sim_time)
            self._set_stage(unit, UnitStage.FURNACE_LOADING)
            self._start(
                "FURNACE_TRANSFER",
                unit,
                "FURNACE_LOAD_TRAY",
                self.durations.furnace_transfer,
            )
            return True
        if not self._loading_batch_ready_to_close():
            return False
        leader = self.units[self._furnace_load_queue[0]]
        if not self._operation_start_allowed(
            "FURNACE_DOOR",
            leader,
            "FURNACE_FRONT_CLOSE",
        ):
            return False
        self.furnace.close_front(now=self.sim_time)
        self._start(
            "FURNACE_DOOR",
            leader,
            "FURNACE_FRONT_CLOSE",
            self.durations.furnace_door,
        )
        return True

    def _dispatch_furnace_unload(self) -> bool:
        if not self._active_batch_units or not self.furnace.ready_to_unload:
            return False
        if not self.furnace.state.rear_door_open and "FURNACE_DOOR" not in self.operations:
            self.furnace.open_rear(now=self.sim_time)
            self._start(
                "FURNACE_DOOR",
                self.units[self._active_batch_units[-1]],
                "FURNACE_REAR_OPEN",
                self.durations.furnace_door,
            )
            return True
        if (
            not self._rear_door_ready
            or "FURNACE_TRANSFER" in self.operations
            or "POST_CAMERA" in self.operations
            or not self._owner_free(TrayOwner.POST_SCAN)
            or not self._target_available(TrayOwner.POST_SCAN)
        ):
            return False
        occupied_layers = [layer for layer in self.furnace.state.layers if layer.tray_id is not None]
        if not occupied_layers:
            return False
        next_layer = max(occupied_layers, key=lambda layer: layer.index)
        assert next_layer.tray_id is not None
        tray_id = next_layer.tray_id
        unit = next(
            self.units[unit_id]
            for unit_id in self._active_batch_units
            if self.units[unit_id].tray_id == tray_id
        )
        if not self._tray_ready(unit):
            return False
        unloaded_tray_id = self.furnace.unload_rear(now=self.sim_time)
        if unloaded_tray_id != tray_id:
            raise RuntimeError("furnace unload order changed during dispatch")
        self._handoff(
            unit,
            TrayOwner.FURNACE,
            TrayOwner.POST_SCAN,
            TrayPhase.POST_BRAZE_INSPECTION,
        )
        self._set_stage(unit, UnitStage.FURNACE_UNLOADING)
        self._start(
            "FURNACE_TRANSFER",
            unit,
            "FURNACE_UNLOAD_TRAY",
            self.durations.furnace_transfer,
        )
        return True

    def _dispatch(self) -> None:
        for _ in range(64):
            changed = False
            changed |= self._dispatch_furnace_unload()
            changed |= self._dispatch_furnace_loading()
            changed |= self._dispatch_transfers()
            changed |= self._dispatch_resources()
            changed |= self._try_start_furnace()
            if not changed:
                return
        raise RuntimeError("V2 dispatch did not converge")

    def tick(self, dt: float) -> dict[str, Any]:
        dt = float(dt)
        if not isfinite(dt) or dt <= 0:
            raise ValueError("tick duration must be finite and positive")
        if self.paused:
            return self.snapshot()
        self.sim_time += dt
        self._advance_operations(dt)
        self.furnace.update(self.sim_time)
        self._dispatch()
        active_wip = sum(tray.owner is not TrayOwner.EMPTY_BUFFER for tray in self.flow.trays)
        self.maximum_wip = max(self.maximum_wip, active_wip)
        return self.snapshot()

    def run_until_complete(self, *, max_sim_time: float, dt: float = 0.05) -> dict[str, Any]:
        maximum = float(max_sim_time)
        if maximum <= self.sim_time:
            raise ValueError("maximum simulation time must be in the future")
        while not self.complete and self.sim_time + 1.0e-12 < maximum:
            self.tick(min(float(dt), maximum - self.sim_time))
        if not self.complete:
            raise TimeoutError(f"V2 line did not complete by {maximum:.1f} simulation seconds")
        return self.snapshot()

    def pause(self) -> None:
        self.paused = True
        self._event("PAUSED")

    def continue_run(self) -> None:
        self.paused = False
        self._event("CONTINUED")

    def reset(self) -> None:
        self.flow.reset(now=0.0)
        self.furnace.reset(now=0.0)
        self.orders.clear()
        self.units.clear()
        self.operations.clear()
        self.events.clear()
        self.sim_time = 0.0
        self.paused = False
        self._order_sequence = 0
        self._active_batch_units.clear()
        self._batch_sequence = 0
        self._batch_recorded = False
        self._furnace_load_queue.clear()
        self._furnace_load_position = 0
        self._loading_batch_started_at = 0.0
        self._rear_door_ready = False
        self.completed_batches.clear()
        self.install_branch_counts = {branch: 0 for branch in InstallBranch}
        self.scheduled_parallel_install_seconds = 0.0
        self.upstream_work_during_brazing_s = 0.0
        self.maximum_wip = 0

    def snapshot(self) -> dict[str, Any]:
        completed_orders = [
            order_id
            for order_id, order in self.orders.items()
            if all(self.units[unit_id].stage is UnitStage.COMPLETE for unit_id in order.unit_ids)
        ]
        tray_states = []
        for tray in self.flow.trays:
            state = tray.as_dict()
            state["stage"] = state.pop("phase")
            tray_states.append(state)
        last_batch = None if not self.completed_batches else dict(self.completed_batches[-1])
        return {
            "schema_version": 2,
            "line": "V2_DUAL_INSTALL",
            # The independent V2 line now gates runtime progress on physical
            # carriers, solved robot paths, tool ownership and constrained
            # in-flight workpiece proxies. It remains a rehearsal because
            # settled product components use a fixed visual pool and standalone
            # segment/fault actors are intentionally disabled.
            "execution_mode": "CONTROL_PLANE_REHEARSAL",
            "physical_execution_complete": False,
            "sim_time": round(self.sim_time, 6),
            "paused": self.paused,
            "complete": self.complete,
            "topology": self.topology.as_dict(),
            "orders": [order.as_dict(self.units) for order in self.orders.values()],
            "completed_orders": completed_orders,
            "units": [unit.as_dict() for unit in self.units.values()],
            "trays": tray_states,
            "operations": {
                resource: {
                    "unit_id": operation.unit_id,
                    "kind": operation.kind,
                    "remaining_s": max(0.0, operation.remaining_s),
                }
                for resource, operation in sorted(self.operations.items())
            },
            "install_branch_counts": {
                branch.value: count for branch, count in self.install_branch_counts.items() if count > 0
            },
            "scheduled_parallel_install_seconds": round(self.scheduled_parallel_install_seconds, 6),
            "furnace": {
                **self.furnace.as_dict(),
                "completed_batches": len(self.completed_batches),
                "last_batch": last_batch,
            },
            "metrics": {
                "upstream_work_during_brazing_s": round(self.upstream_work_during_brazing_s, 6),
                "maximum_wip": self.maximum_wip,
            },
            "events": list(self.events),
        }


__all__ = ["DualLineRuntime", "UnitStage", "V2OrderState", "V2UnitState"]
