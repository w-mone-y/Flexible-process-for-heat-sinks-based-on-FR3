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

from ..flexible import build_inline_plan
from ..flexible.models import ProcessPlan
from ..recovery.fault_models import RecoveryStatus
from .faults import V2FaultController
from .camera_coordination import CameraCoordinationPolicy, CameraReviewReason
from .dispatch import (
    DualInstallDispatcher,
    InstallBranch,
    InstallRequest,
    InstallResourceState,
)
from .furnace import BatchRecipe, FurnacePhase, ThroughBatchFurnace
from .inspection_windows import (
    Arm3InspectionWindow,
    InspectionWindowRequest,
    schedule_arm3_inspection_windows,
)
from .process_geometry import V2_MAX_PRODUCT_HEIGHT_M, V2ProcessGeometry
from .topology import DualLineTopology
from .tray_flow import TrayFlowController, TrayOwner, TrayPhase
from ..events import EventType
from ..twin import DecisionEvent, DigitalTwinSnapshot


class UnitStage(str, Enum):
    QUEUED = "QUEUED"
    BASE_LOADING = "BASE_LOADING"
    WAITING_S2A = "WAITING_S2A"
    DISPENSING = "DISPENSING"
    WAITING_S2B = "WAITING_S2B"
    MATERIAL_INSPECTION = "MATERIAL_INSPECTION"
    WAITING_BRAZING_REVIEW = "WAITING_BRAZING_REVIEW"
    BRAZING_REVIEW = "BRAZING_REVIEW"
    WAITING_INSTALL = "WAITING_INSTALL"
    FIN_INSTALLATION = "FIN_INSTALLATION"
    WAITING_FINS_REVIEW = "WAITING_FINS_REVIEW"
    FINS_REVIEW = "FINS_REVIEW"
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
    product_id: str
    process_geometry: V2ProcessGeometry
    comb_module: str
    target_clamping_force_n: float
    batch_recipe: BatchRecipe
    route_strategy: str
    urgent: bool = False
    stage: UnitStage = UnitStage.QUEUED
    tray_id: str | None = None
    branch: InstallBranch | None = None
    fins_installed: int = 0
    stage_started_at: float = 0.0
    completed_at: float | None = None
    buffer_owner: TrayOwner | None = None
    furnace_layer: int | None = None
    rework_fin_index: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "order_id": self.order_id,
            "preset": self.preset,
            "product_id": self.product_id,
            "fin_count": self.fin_count,
            "path_count": len(self.process_geometry.brazing_paths),
            "comb_module": self.comb_module,
            "target_clamping_force_n": self.target_clamping_force_n,
            "recipe": self.batch_recipe.name,
            "material_system": self.batch_recipe.material_system,
            "route_strategy": self.route_strategy,
            "priority": self.priority,
            "due_at": self.due_at,
            "urgent": self.urgent,
            "stage": self.stage.value,
            "tray_id": self.tray_id,
            "branch": None if self.branch is None else self.branch.value,
            "fins_installed": self.fins_installed,
            "furnace_layer": self.furnace_layer,
            "completed_at": self.completed_at,
            "rework_fin_index": self.rework_fin_index,
        }


@dataclass(slots=True)
class V2OrderState:
    order_id: str
    preset: str
    priority: int
    unit_ids: tuple[str, ...]
    inserted_at: float
    product_id: str
    route_strategy: str
    mode: str
    due_at: float | None = None
    urgent: bool = False

    def as_dict(self, units: dict[str, V2UnitState]) -> dict[str, object]:
        complete = all(units[unit_id].stage is UnitStage.COMPLETE for unit_id in self.unit_ids)
        return {
            "order_id": self.order_id,
            "preset": self.preset,
            "product_id": self.product_id,
            "route_strategy": self.route_strategy,
            "mode": self.mode,
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
    duration_s: float
    recovery: bool = False
    recovery_strategy: str = ""
    recovery_fault_type: str = ""
    recovery_target_index: int | None = None
    fault_hold_remaining_s: float = 0.0
    hold_fault_ids: tuple[str, ...] = ()
    manual_hold_fault_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _RecoveryWork:
    effort_factor: float
    strategy: str
    fault_type: str
    target_index: int | None


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

    def install_transfer_allowed(self, unit_id: str, resource: str) -> bool: ...


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


# Stage progression index, used to tell forward motion from a recovery rewind.
_STAGE_ORDER: dict[UnitStage, int] = {stage: index for index, stage in enumerate(UnitStage)}


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
        self._output_gate_open = False
        self._output_gate_ready = False
        self.completed_batches: list[dict[str, Any]] = []
        self.install_branch_counts = {branch: 0 for branch in InstallBranch}
        self.scheduled_parallel_install_seconds = 0.0
        self.upstream_work_during_brazing_s = 0.0
        self.robot_transport_overlap_s = 0.0
        self.s1_s2a_dual_occupancy_s = 0.0
        self.preposition_seconds = {resource: 0.0 for resource in ("ARM1", "ARM2", "ARM3")}
        self.arm3_inspection_reservation_wait_s = 0.0
        self._arm3_reservation_blocked = False
        self.maximum_wip = 0
        # Disturbance flexibility: fault injection, recovery planning and
        # resource isolation.  Holds no MuJoCo reference, so the logical runtime
        # stays testable headless.
        self.faults = V2FaultController()
        self.camera_coordination = CameraCoordinationPolicy()
        # unit_id -> duration multiplier for its next operation (rework surcharge)
        self._rework_effort: dict[str, _RecoveryWork] = {}
        self._execution_gate: RuntimeExecutionGate | None = None

    def set_execution_gate(self, gate: RuntimeExecutionGate | None) -> None:
        """Bind physical readiness feedback, or restore logical-only mode."""

        self._execution_gate = gate

    @property
    def output_gate_open(self) -> bool:
        """Desired physical state of the finished-output gate."""

        return self._output_gate_open

    @property
    def complete(self) -> bool:
        return bool(self.units) and all(unit.stage is UnitStage.COMPLETE for unit in self.units.values())

    @property
    def next_order_id(self) -> str:
        """Stable identifier offered to control surfaces before submission."""

        return f"V2_ORDER_{self._order_sequence + 1:03d}"

    def submit_order(
        self,
        preset: str,
        *,
        order_id: str | None = None,
        quantity: int = 1,
        priority: int = 10,
        due_at: float | None = None,
        urgent: bool = False,
        route_strategy: str = "STANDARD",
    ) -> V2OrderState:
        preset = str(preset).strip().upper()
        if preset not in {"A", "B", "C", "D"}:
            raise ValueError("V2 preset must be A, B, C or D")
        identifier = str(order_id or "").strip()
        if not identifier:
            identifier = self.next_order_id
        plan = build_inline_plan(
            preset=preset,
            order_id=identifier,
            quantity=int(quantity),
            priority=int(priority),
            route_strategy=str(route_strategy).strip().upper() or "STANDARD",
        )
        return self.submit_plan(plan, due_at=due_at, urgent=urgent)

    def submit_plan(
        self,
        plan: ProcessPlan,
        *,
        due_at: float | None = None,
        urgent: bool = False,
        dispatch: bool = True,
    ) -> V2OrderState:
        """Bind one validated ``ProcessPlan`` to V2 logical and physical state."""

        if not 1 <= int(plan.quantity) <= 3:
            raise ValueError("one V2 order may contain one to three units")
        if plan.order.priority < 0:
            raise ValueError("priority must be non-negative")
        if due_at is not None and not isfinite(float(due_at)):
            raise ValueError("due time must be finite")
        geometry = V2ProcessGeometry.from_plan(plan)
        batch_recipe = BatchRecipe(
            name=plan.recipe.name,
            material_system=plan.product.material_system,
            peak_c=float(plan.recipe.peak_c),
            soak_seconds=float(plan.recipe.soak_seconds),
            # This field describes the common V2 rack envelope so compatible
            # A/B/C/custom products can share a batch after individual height
            # validation above.
            maximum_product_height_m=V2_MAX_PRODUCT_HEIGHT_M,
        )
        self._order_sequence += 1
        identifier = str(plan.order.order_id).strip() or f"V2_ORDER_{self._order_sequence:03d}"
        if identifier in self.orders:
            raise ValueError(f"duplicate V2 order id: {identifier}")
        unit_ids = tuple(f"{identifier}_UNIT_{index:02d}" for index in range(1, int(plan.quantity) + 1))
        for unit_id in unit_ids:
            self.units[unit_id] = V2UnitState(
                unit_id=unit_id,
                order_id=identifier,
                preset=plan.product.preset,
                fin_count=len(plan.fin_targets),
                priority=int(plan.order.priority),
                due_at=None if due_at is None else float(due_at),
                product_id=plan.product.product_id,
                process_geometry=geometry,
                comb_module=plan.fixture_module.name,
                target_clamping_force_n=float(plan.product.target_clamping_force_n),
                batch_recipe=batch_recipe,
                route_strategy=plan.route_strategy.value,
                urgent=bool(urgent),
            )
        order = V2OrderState(
            order_id=identifier,
            preset=plan.product.preset,
            priority=int(plan.order.priority),
            unit_ids=unit_ids,
            inserted_at=self.sim_time,
            product_id=plan.product.product_id,
            route_strategy=plan.route_strategy.value,
            mode="custom" if plan.product.preset == "CUSTOM" else "preset",
            due_at=None if due_at is None else float(due_at),
            urgent=bool(urgent),
        )
        self.orders[identifier] = order
        self._event(
            "ORDER_QUEUED",
            order_id=identifier,
            preset=plan.product.preset,
            product_id=plan.product.product_id,
            quantity=plan.quantity,
            priority=int(plan.order.priority),
            fin_count=len(plan.fin_targets),
            path_count=len(plan.brazing_paths),
            comb_module=plan.fixture_module.name,
            route_strategy=plan.route_strategy.value,
            due_at=order.due_at,
            urgent=order.urgent,
        )
        if dispatch:
            self._dispatch()
        return order

    def _event(self, event_type: str, **payload: Any) -> None:
        self.events.append({"time": round(self.sim_time, 6), "type": event_type, **payload})

    @staticmethod
    def _camera_review_reason(unit: V2UnitState) -> CameraReviewReason | None:
        strategy = str(unit.route_strategy).upper()
        if strategy == "HIGH_RELIABILITY":
            return CameraReviewReason.HIGH_RELIABILITY
        if strategy == "FIRST_ARTICLE" and unit.unit_id.endswith("_UNIT_01"):
            return CameraReviewReason.FIRST_ARTICLE
        return None

    def _unit_requires_camera_review(self, unit: V2UnitState, inspection_kind: str) -> bool:
        if inspection_kind not in {"MATERIAL_INSPECTION", "PRE_BRAZE_INSPECTION"}:
            return False
        return self._camera_review_reason(unit) is not None

    def camera_review_required(self, unit_id: str, inspection_kind: str) -> bool:
        """Whether an active Arm3 operation is the S3B route review."""

        unit = self.units.get(str(unit_id))
        if unit is None or not self._unit_requires_camera_review(unit, inspection_kind):
            return False
        return (inspection_kind == "MATERIAL_INSPECTION" and unit.stage is UnitStage.BRAZING_REVIEW) or (
            inspection_kind == "PRE_BRAZE_INSPECTION" and unit.stage is UnitStage.FINS_REVIEW
        )

    def _start_camera_review(self, unit: V2UnitState, inspection_kind: str) -> None:
        reason = self._camera_review_reason(unit)
        if reason is None:
            return
        request = self.camera_coordination.request(
            unit_id=unit.unit_id,
            inspection_kind=inspection_kind,
            station_id="S3B_ARM3_INSTALL",
            reason=reason,
            now=self.sim_time,
        )
        if request.status == "QUEUED":
            self.camera_coordination.mark_started(unit.unit_id, inspection_kind, self.sim_time)

    def camera_coordination_snapshot(self) -> dict[str, Any]:
        arm3_online = self._resource_online("ARM3")
        blocked_reason = "Arm3相机离线，工件原位阻塞，等待恢复"
        stations = [
            {
                "station_id": "S2B_MATERIAL_INSPECTION",
                "primary_camera": "ARM3_CAMERA",
                "secondary_camera": None,
                "inspection_kinds": ["MATERIAL_INSPECTION"],
            },
            {
                "station_id": "S4_PRE_BRAZE_INSPECTION",
                "primary_camera": "ARM3_CAMERA",
                "secondary_camera": None,
                "inspection_kinds": ["PRE_BRAZE_INSPECTION"],
            },
            {
                "station_id": "S3B_ARM3_INSTALL",
                "primary_camera": "ARM3_CAMERA",
                "secondary_camera": None,
                "inspection_kinds": ["MATERIAL_INSPECTION", "PRE_BRAZE_INSPECTION"],
            },
            {
                "station_id": "POST_BRAZE_SCAN",
                "primary_camera": "POST_CAMERA",
                "secondary_camera": None,
                "inspection_kinds": ["POST_BRAZE_INSPECTION"],
            },
        ]
        for station in stations:
            if station["primary_camera"] == "ARM3_CAMERA" and not arm3_online:
                station["status"] = "BLOCKED"
                station["reason_zh"] = blocked_reason
            else:
                station["status"] = "READY"
                station["reason_zh"] = ""
        active_units = [
            unit
            for unit in self.units.values()
            if self._camera_review_reason(unit) is not None and unit.stage is not UnitStage.COMPLETE
        ]
        snapshot = self.camera_coordination.snapshot()
        snapshot["stations"] = stations
        snapshot["active_plan"] = {
            "status": "RUNNING" if snapshot["pending_reviews"] or active_units else "STANDBY",
            "unit_ids": [unit.unit_id for unit in active_units],
            "policy": "SINGLE_ARM3_CAMERA_WITH_S3B_CLOSEUPS",
        }
        return snapshot

    def _set_stage(self, unit: V2UnitState, stage: UnitStage) -> None:
        unit.stage = stage
        unit.stage_started_at = self.sim_time
        self._event(
            "UNIT_STAGE",
            unit_id=unit.unit_id,
            tray_id=unit.tray_id,
            stage=stage.value,
        )
        # Recovery is completed explicitly by the follow-up camera operation;
        # mere forward motion after a repair is not proof of quality.

    def _resource_online(self, resource: str) -> bool:
        """False while a resource is isolated by an ARM_UNAVAILABLE fault."""

        return self.faults.resource_available(resource)

    def recovery_operation_authorized(self, resource: str, unit_id: str, kind: str) -> bool:
        """Authorize only the physical operation created by an applied rollback."""

        work = self._rework_effort.get(str(unit_id))
        if work is None:
            return False
        expected = {
            "LOCAL_BRAZING_REWORK": ("ARM2", "DISPENSING"),
            "FIN_REINSTALL": ("ARM3", "INSTALL_FIN"),
        }.get(work.strategy)
        return expected == (str(resource).upper(), str(kind).upper())

    def _start(self, resource: str, unit: V2UnitState, kind: str, duration: float) -> bool:
        """Begin an operation, or decline when the resource is unavailable.

        Returns False when a fault has isolated the resource.  Declining rather
        than raising keeps the guard fail-safe: several dispatch branches call
        ``_start`` without first consulting ``_operation_start_allowed``, and an
        isolated mechanism must stall the line, not crash the tick.
        """

        if resource in self.operations:
            raise RuntimeError(f"resource already busy: {resource}")
        # Every operation, including ordinary robot work, must cross the same
        # execution-authority seam.  Previously only one furnace-door branch
        # called ``_operation_start_allowed``; an external scheduler therefore
        # could not prevent Arm1/2/3 from starting work on its own.  Keeping the
        # check here makes ``_start`` the single physical dispatch choke point.
        if not self._operation_start_allowed(resource, unit, kind):
            return False
        # A unit carrying pending rework performs the next operation at rework
        # effort, then the surcharge is consumed.
        recovery_work = self._rework_effort.pop(unit.unit_id, None)
        recovery = recovery_work is not None
        if recovery_work is not None:
            duration = float(duration) * float(recovery_work.effort_factor)
            self._event(
                "REWORK_EFFORT_APPLIED",
                resource=resource,
                unit_id=unit.unit_id,
                kind=kind,
                factor=recovery_work.effort_factor,
                strategy=recovery_work.strategy,
                target_index=recovery_work.target_index,
            )
        hold = self.faults.take_hold(unit.unit_id)
        if hold > 0.0:
            duration = float(duration) + hold
            self._event(
                "FAULT_HOLD_APPLIED",
                resource=resource,
                unit_id=unit.unit_id,
                kind=kind,
                seconds=hold,
                applied_to="NEXT_OPERATION",
            )
        self.operations[resource] = _Operation(
            resource=resource,
            unit_id=unit.unit_id,
            kind=kind,
            remaining_s=float(duration),
            started_at=self.sim_time,
            duration_s=float(duration),
            recovery=recovery,
            recovery_strategy="" if recovery_work is None else recovery_work.strategy,
            recovery_fault_type="" if recovery_work is None else recovery_work.fault_type,
            recovery_target_index=(None if recovery_work is None else recovery_work.target_index),
        )
        if self.camera_review_required(unit.unit_id, kind):
            self._start_camera_review(unit, kind)
        self._event("OPERATION_STARTED", resource=resource, unit_id=unit.unit_id, kind=kind)
        return True

    def _resource_available_at(self, resource: str) -> float:
        operation = self.operations.get(resource)
        if operation is not None and operation.manual_hold_fault_ids:
            return float("inf")
        return (
            self.sim_time
            if operation is None
            else self.sim_time + operation.remaining_s + operation.fault_hold_remaining_s
        )

    def _waiting_units(self, *stages: UnitStage) -> list[V2UnitState]:
        allowed = set(stages)
        return sorted(
            (
                unit
                for unit in self.units.values()
                if unit.stage in allowed
                and not self._unit_waiting_for_manual_review(unit.unit_id)
                and not self._unit_waiting_for_recovery_transport(unit.unit_id)
            ),
            key=lambda unit: (
                -int(unit.urgent),
                -unit.priority,
                float("inf") if unit.due_at is None else unit.due_at,
                unit.stage_started_at,
                unit.unit_id,
            ),
        )

    def _unit_waiting_for_manual_review(self, unit_id: str) -> bool:
        return any(
            plan.unit_id == unit_id and plan.status is RecoveryStatus.MANUAL_REVIEW
            for plan in self.faults.plans.values()
        )

    def _unit_waiting_for_recovery_transport(self, unit_id: str) -> bool:
        """Keep a quality-fault unit out of normal dispatch until it returns.

        A detected defect can coincide with a full S2A or S3B station.  The
        recovery planner must wait for that physical reservation rather than
        allowing the unit's old stage (``MATERIAL_INSPECTION`` or
        ``PRE_BRAZE_INSPECTION``) to be dispatched as a fresh forward task.
        """

        return any(
            plan.unit_id == unit_id
            and plan.status is RecoveryStatus.RUNNING
            and not plan.rollback_applied
            and plan.strategy in {"LOCAL_BRAZING_REWORK", "FIN_REINSTALL"}
            for plan in self.faults.plans.values()
        )

    def _s2a_inspection_clearance_required(self) -> bool:
        """Keep S2A clear until the pallet at S2B passes material inspection.

        S2A is both the forward dispensing station and the only physical
        return destination for local brazing rework.  Releasing the next S1
        pallet while the previous pallet is still being inspected can fill
        S2A just before a defect is detected, creating a circular wait.  The
        reservation is released in the same dispatch cycle that inspection
        succeeds, so a passing pallet does not introduce an extra idle delay.
        """

        for unit in self.units.values():
            if unit.tray_id is None:
                continue
            tray = self.flow.get(unit.tray_id)
            if tray.owner is TrayOwner.S2B and unit.stage in {
                UnitStage.WAITING_S2B,
                UnitStage.MATERIAL_INSPECTION,
            }:
                return True

        # Keep the explicit fault-plan checks as a fail-safe for externally
        # injected detections whose stage update may arrive one tick later.
        for defect in self.faults.physical_faults.values():
            if defect.status not in {"MANIFESTED", "DETECTED"}:
                continue
            if defect.fault_type.value not in {"BRAZING_MISSING", "BRAZING_PATH_DEVIATION"}:
                continue
            unit = self.units.get(defect.unit_id)
            if unit is None or unit.tray_id is None:
                continue
            if self.flow.get(unit.tray_id).owner is TrayOwner.S2B:
                return True
        return any(
            plan.strategy == "LOCAL_BRAZING_REWORK"
            and plan.status is RecoveryStatus.RUNNING
            and not plan.rollback_applied
            and plan.unit_id is not None
            and self.units.get(plan.unit_id) is not None
            and self.units[plan.unit_id].tray_id is not None
            and self.flow.get(self.units[plan.unit_id].tray_id).owner is TrayOwner.S2B
            for plan in self.faults.plans.values()
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

    def _install_transfer_allowed(self, unit: V2UnitState) -> bool:
        if (
            unit.branch is InstallBranch.ARM3_B
            and not self.faults.arm3_rework_station_available_to(unit.unit_id)
        ):
            return False
        if self._execution_gate is None or unit.branch is None:
            return True
        callback = getattr(self._execution_gate, "install_transfer_allowed", None)
        if callback is None:
            return True
        resource = "ARM1" if unit.branch is InstallBranch.ARM1_A else "ARM3"
        return bool(callback(unit.unit_id, resource))

    def _estimated_tray_ready_in(self, unit: V2UnitState, owner: TrayOwner) -> float:
        if unit.tray_id is None:
            return float("inf")
        if self._execution_gate is None or self._tray_ready(unit):
            return 0.0
        callback = getattr(self._execution_gate, "estimated_tray_ready_in", None)
        if callback is None:
            # A gate without a forecast remains fail-safe: reserve immediately
            # rather than allowing new Arm3 work to delay an incoming check.
            return 0.0
        estimate = float(callback(unit.tray_id, owner))
        return estimate if isfinite(estimate) and estimate >= 0.0 else float("inf")

    def arm3_inspection_windows(self) -> tuple[Arm3InspectionWindow, ...]:
        """Predict Arm3 checks while preserving physical arrival authority."""

        stage_specs = {
            UnitStage.WAITING_S2B: (
                "MATERIAL_INSPECTION",
                TrayOwner.S2B,
                self.durations.material_inspection,
                "托盘预计到达S2B，预留Arm3钎料检测时间窗",
            ),
            UnitStage.MATERIAL_INSPECTION: (
                "MATERIAL_INSPECTION",
                TrayOwner.S2B,
                self.durations.material_inspection,
                "托盘正在进入S2B，预留Arm3钎料检测时间窗",
            ),
            UnitStage.WAITING_S4: (
                "PRE_BRAZE_INSPECTION",
                TrayOwner.S4,
                self.durations.pre_braze_inspection,
                "托盘预计到达S4，预留Arm3焊前检测时间窗",
            ),
            UnitStage.PRE_BRAZE_INSPECTION: (
                "PRE_BRAZE_INSPECTION",
                TrayOwner.S4,
                self.durations.pre_braze_inspection,
                "托盘正在进入S4，预留Arm3焊前检测时间窗",
            ),
        }
        requests: list[InspectionWindowRequest] = []
        candidates = self._waiting_units(*stage_specs)
        for unit in candidates:
            kind, owner, duration, reason = stage_specs[unit.stage]
            incoming_stage = unit.stage in {UnitStage.WAITING_S2B, UnitStage.WAITING_S4}
            if incoming_stage and not (
                self._owner_free(owner) and self._target_available(owner)
            ):
                # A pallet blocked outside the inspection station is not an
                # actionable Arm3 reservation. Reserving it immediately can
                # prevent Arm3 from finishing the S3B pallet that must leave
                # before a defective S4 pallet can return for rework, creating
                # a closed wait cycle. Once the destination is physically
                # available this candidate reappears and regains inspection
                # priority at the next non-preemptible-fin boundary.
                continue
            active = self.operations.get("ARM3")
            if active is not None and active.unit_id == unit.unit_id and active.kind == kind:
                continue
            ready_in = self._estimated_tray_ready_in(unit, owner)
            requests.append(
                InspectionWindowRequest(
                    unit_id=unit.unit_id,
                    inspection_kind=kind,
                    source_stage=unit.stage.value,
                    ready_at=self.sim_time + ready_in,
                    duration_s=duration,
                    reason_zh=reason,
                )
            )
        return schedule_arm3_inspection_windows(
            requests,
            arm3_available_at=self._resource_available_at("ARM3"),
        )

    def _arm3_fin_fits_before_inspection(self, unit: V2UnitState) -> bool:
        committed_callback = (
            None
            if self._execution_gate is None
            else getattr(self._execution_gate, "arm3_fin_operation_committed", None)
        )
        if committed_callback is not None and committed_callback(unit.unit_id):
            # The unified DAG grants one fin at a time. Once that boundary has
            # been crossed, physical execution must finish the authorized fin;
            # the next scheduler boundary will give inspection first priority.
            return True
        finish_at = self.sim_time + self.durations.arm3_fin
        for window in self.arm3_inspection_windows():
            if self.sim_time < window.end_at and finish_at > window.start_at:
                self._arm3_reservation_blocked = True
                return False
        return True

    def _arm1_tail_fin_candidate(self) -> V2UnitState | None:
        """Return future Arm1 fin work once the accepted base wave is complete.

        This is a tool-preparation forecast, not an install-branch reservation.
        An unassigned upstream unit may therefore justify mounting the gripper,
        but the normal dispatcher remains solely responsible for choosing Arm1
        or Arm3 after inspection.  A newly inserted base order immediately
        closes this forecast and lets the higher-priority S1 suction intent win.
        """

        if any(unit.stage in {UnitStage.QUEUED, UnitStage.BASE_LOADING} for unit in self.units.values()):
            return None
        upstream = self._waiting_units(
            UnitStage.WAITING_S2A,
            UnitStage.DISPENSING,
            UnitStage.WAITING_S2B,
            UnitStage.MATERIAL_INSPECTION,
            UnitStage.WAITING_BRAZING_REVIEW,
            UnitStage.BRAZING_REVIEW,
            UnitStage.WAITING_INSTALL,
            UnitStage.FIN_INSTALLATION,
        )
        candidates = [
            unit
            for unit in upstream
            if unit.branch is not InstallBranch.ARM3_B
            and (unit.fins_installed < unit.fin_count or unit.rework_fin_index is not None)
        ]
        return next(
            (unit for unit in candidates if unit.branch is InstallBranch.ARM1_A),
            next(iter(candidates), None),
        )

    def next_s1_base_ready_in(self) -> float:
        """Forecast the next admitted blank without reserving Arm1.

        The logical S1 owner changes before the authored carrier has always
        settled.  Looking only at immediately executable PICK_BASE tasks can
        therefore miss a blank that will arrive as soon as the current S1
        tray advances to a clearing S2A lane.  The estimate is advisory and
        never moves a tray or bypasses ownership checks.
        """

        base_loading = self._waiting_units(UnitStage.BASE_LOADING)
        for unit in base_loading:
            if self._tray_ready(unit):
                return 0.0
            return self._estimated_tray_ready_in(unit, TrayOwner.S1)
        if not self._waiting_units(UnitStage.QUEUED):
            return float("inf")
        s1_blockers = self._waiting_units(UnitStage.WAITING_S2A)
        if not s1_blockers:
            return 0.0 if self._owner_free(TrayOwner.S1) else float("inf")
        if self._owner_free(TrayOwner.S2A) and self._target_available(TrayOwner.S2A):
            return 0.0
        # S2A may already be logically free while its previous tray is still
        # completing the authored S2A->S2B rail.  Include that measured
        # remainder rather than treating the next S1 blank as unavailable.
        downstream = self._waiting_units(UnitStage.WAITING_S2B, UnitStage.MATERIAL_INSPECTION)
        estimates = [
            self._estimated_tray_ready_in(unit, TrayOwner.S2B)
            for unit in downstream
            if unit.tray_id is not None
        ]
        finite = [value for value in estimates if isfinite(value)]
        return min(finite, default=float("inf"))

    def prepositioning_snapshot(self) -> dict[str, dict[str, str]]:
        """Return safe robot approach intents while the target tray is moving.

        An intent is deliberately weaker than an operation: it never changes
        task status, reserves tray ownership, or authorizes process contact.
        The MuJoCo actor may only use it to reach an authored aerial approach
        pose.  ``_tray_ready`` remains the single gate for starting the actual
        operation after the carrier has arrived and settled.
        """

        if self.paused or self._execution_gate is None or not self.faults.cell_available():
            return {}
        rules = (
            (
                "ARM1",
                UnitStage.BASE_LOADING,
                "BASE_LOADING",
                "S1_BASE_LOADING",
                "空托盘运输至S1时提前完成吸盘准备并到达基板安全接近位",
                None,
            ),
            (
                "ARM2",
                UnitStage.DISPENSING,
                "DISPENSING",
                "S2A_DISPENSING",
                "托盘运输至S2A时提前到达安全接近位",
                None,
            ),
            (
                "ARM3",
                UnitStage.MATERIAL_INSPECTION,
                "MATERIAL_INSPECTION",
                "S2B_MATERIAL_INSPECTION",
                "托盘运输至S2B时提前到达相机安全位",
                None,
            ),
            (
                "ARM3",
                UnitStage.BRAZING_REVIEW,
                "MATERIAL_INSPECTION",
                "S3B_ARM3_INSTALL",
                "托盘返往S3B近景复核时提前到达相机安全位",
                InstallBranch.ARM3_B,
            ),
            (
                "ARM3",
                UnitStage.PRE_BRAZE_INSPECTION,
                "PRE_BRAZE_INSPECTION",
                "S4_PRE_BRAZE_INSPECTION",
                "托盘合流至S4时提前到达焊前检测安全位",
                None,
            ),
            (
                "ARM3",
                UnitStage.FINS_REVIEW,
                "PRE_BRAZE_INSPECTION",
                "S3B_ARM3_INSTALL",
                "托盘返往S3B翅片复核时提前到达相机安全位",
                InstallBranch.ARM3_B,
            ),
            (
                "ARM1",
                UnitStage.FIN_INSTALLATION,
                "INSTALL_FIN",
                "S3A_ARM1_INSTALL",
                "托盘运输至S3A时提前完成夹爪准备并到达翅片原料安全位",
                InstallBranch.ARM1_A,
            ),
            (
                "ARM3",
                UnitStage.FIN_INSTALLATION,
                "INSTALL_FIN",
                "S3B_ARM3_INSTALL",
                "托盘运输至S3B时提前到达翅片原料安全位",
                InstallBranch.ARM3_B,
            ),
        )
        intents: dict[str, dict[str, str]] = {}
        for resource, stage, operation_kind, station_id, reason_zh, branch in rules:
            if resource in self.operations or not self._resource_online(resource):
                continue
            if resource in intents:
                continue
            if (
                resource == "ARM1"
                and operation_kind == "INSTALL_FIN"
                and self.next_s1_base_ready_in() <= 12.0
                and not bool(
                    getattr(
                        self._execution_gate,
                        "arm1_fin_preposition_allowed",
                        lambda: False,
                    )()
                )
            ):
                # Standalone V2 keeps the conservative S1 lookahead.  Under
                # the unified facade, the single DAG authority may explicitly
                # release the fin wave even while another admitted blank is
                # visible; in that case physical prepositioning must follow
                # the scheduler instead of re-applying a conflicting rule.
                continue
            physically_ready_work = any(
                unit.tray_id is not None
                and self._tray_ready(unit)
                and (
                    (resource == "ARM1" and unit.stage is UnitStage.BASE_LOADING)
                    or (resource == "ARM2" and unit.stage is UnitStage.DISPENSING)
                    or (
                        resource == "ARM3"
                        and unit.stage
                        in {
                            UnitStage.MATERIAL_INSPECTION,
                            UnitStage.PRE_BRAZE_INSPECTION,
                            UnitStage.BRAZING_REVIEW,
                            UnitStage.FINS_REVIEW,
                        }
                    )
                    or (
                        resource == "ARM1"
                        and unit.stage is UnitStage.FIN_INSTALLATION
                        and unit.branch is InstallBranch.ARM1_A
                    )
                    or (
                        resource == "ARM3"
                        and unit.stage is UnitStage.FIN_INSTALLATION
                        and unit.branch is InstallBranch.ARM3_B
                    )
                )
                for unit in self.units.values()
            )
            if physically_ready_work:
                continue
            candidate = next(
                (
                    unit
                    for unit in self._waiting_units(stage)
                    if unit.tray_id is not None
                    and (branch is None or unit.branch is branch)
                    and not self._tray_ready(unit)
                ),
                None,
            )
            if candidate is None:
                continue
            intents[resource] = {
                "unit_id": candidate.unit_id,
                "operation_kind": operation_kind,
                "station_id": station_id,
                "reason_zh": reason_zh,
            }
        if "ARM1" not in intents and "ARM1" not in self.operations and self._resource_online("ARM1"):
            tail_candidate = self._arm1_tail_fin_candidate()
            if tail_candidate is not None:
                intents["ARM1"] = {
                    "unit_id": tail_candidate.unit_id,
                    "operation_kind": "INSTALL_FIN",
                    "station_id": "S3A_ARM1_INSTALL",
                    "reason_zh": ("最后一块基板已安装，提前切换夹爪；不提前锁定安装支路或接触托盘"),
                }
        return intents

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
        # A resource isolated by a fault cannot start anything.  Checking it here
        # covers every physical start through one choke point, rather than
        # relying on each dispatch branch to remember the guard.
        if not self._resource_online(resource):
            return False
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

    def _arm3_downstream_blocking_estimate(self) -> float:
        """Estimate congestion Arm3 would feed into the shared merge/S4 path."""

        congested = sum(
            unit.stage
            in {
                UnitStage.WAITING_MERGE,
                UnitStage.MERGING,
                UnitStage.WAITING_S4,
                UnitStage.PRE_BRAZE_INSPECTION,
                UnitStage.WAITING_BUFFER,
            }
            for unit in self.units.values()
        )
        return congested * (self.durations.merge + self.durations.pre_braze_inspection)

    def _assign_branch(self, unit: V2UnitState) -> InstallBranch:
        camera_review_required = self._camera_review_reason(unit) is not None
        arm3_fault_demonstration_required = self.faults.claim_arm3_fin_install(unit.unit_id)
        force_arm3 = camera_review_required or arm3_fault_demonstration_required
        request = InstallRequest(
            tray_id=unit.tray_id or unit.unit_id,
            fin_count=unit.fin_count,
            ready_at=self.sim_time,
            due_at=unit.due_at,
            priority=unit.priority,
        )
        arm3_reservations = tuple(
            (window.start_at, window.end_at) for window in self.arm3_inspection_windows()
        )
        decision = self.dispatcher.assign(
            request,
            (
                InstallResourceState(
                    InstallBranch.ARM1_A,
                    self._resource_available_at("ARM1"),
                    self.durations.arm1_fin,
                    queued_fins=self._queued_fins(InstallBranch.ARM1_A, excluding=unit.unit_id),
                    enabled=self._resource_online("ARM1") and not force_arm3,
                ),
                InstallResourceState(
                    InstallBranch.ARM3_B,
                    self._resource_available_at("ARM3"),
                    self.durations.arm3_fin,
                    queued_fins=self._queued_fins(InstallBranch.ARM3_B, excluding=unit.unit_id),
                    inspection_reservations=arm3_reservations,
                    downstream_blocking_s=self._arm3_downstream_blocking_estimate(),
                    enabled=(
                        self._resource_online("ARM3")
                        and self.faults.arm3_rework_station_available_to(unit.unit_id)
                    ),
                ),
            ),
        )
        selected_branch = decision.branch
        unit.branch = selected_branch
        self.install_branch_counts[selected_branch] += 1
        selected_candidate = decision.candidates[selected_branch]
        self._event(
            "INSTALL_ASSIGNED",
            unit_id=unit.unit_id,
            tray_id=unit.tray_id,
            branch=selected_branch.value,
            explanation_zh=(
                "高可靠路线要求Arm3在同一支路完成安装与近景复核"
                if camera_review_required
                else (
                    "故障验收要求由Arm3执行本次翅片安装"
                    if arm3_fault_demonstration_required
                    else decision.explanation_zh
                )
            ),
            selected_cost=selected_candidate.cost,
            arm3_activated=decision.arm3_activated,
            arm3_expected_gain_s=decision.arm3_expected_gain_s,
            arm3_inspection_penalty_s=decision.arm3_inspection_penalty_s,
            arm3_blocking_penalty_s=decision.arm3_blocking_penalty_s,
            arm3_net_gain_s=decision.arm3_net_gain_s,
            activation_reason_zh=decision.activation_reason_zh,
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
        return selected_branch

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
        if (
            replacement is InstallBranch.ARM3_B
            and not self.faults.arm3_rework_station_available_to(unit.unit_id)
        ):
            return False
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
        if operation.recovery:
            self.faults.mark_repaired(unit.unit_id, self.sim_time)
            self._event(
                "FAULT_REPAIRED",
                unit_id=unit.unit_id,
                kind=operation.kind,
                resource=operation.resource,
            )
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
            self.faults.complete_recovery(unit.unit_id, self.sim_time)
            if unit.stage is UnitStage.BRAZING_REVIEW:
                self.camera_coordination.complete(
                    unit.unit_id,
                    operation.kind,
                    now=self.sim_time,
                    result="PASS",
                )
                self._set_stage(unit, UnitStage.WAITING_INSTALL)
            else:
                self._assign_branch(unit)
                next_stage = (
                    UnitStage.WAITING_BRAZING_REVIEW
                    if self._unit_requires_camera_review(unit, operation.kind)
                    else UnitStage.WAITING_INSTALL
                )
                self._set_stage(unit, next_stage)
        elif operation.kind == "INSTALL_FIN":
            installed_index = unit.rework_fin_index or (unit.fins_installed + 1)
            if unit.rework_fin_index is None:
                unit.fins_installed += 1
            else:
                unit.rework_fin_index = None
            self._event(
                "FIN_INSTALLED",
                unit_id=unit.unit_id,
                fin_index=installed_index,
                fin_count=unit.fin_count,
                branch=None if unit.branch is None else unit.branch.value,
            )
            if unit.fins_installed >= unit.fin_count:
                next_stage = (
                    UnitStage.WAITING_FINS_REVIEW
                    if self._unit_requires_camera_review(unit, "PRE_BRAZE_INSPECTION")
                    else UnitStage.WAITING_MERGE
                )
                self._set_stage(unit, next_stage)
            else:
                self._set_stage(unit, UnitStage.FIN_INSTALLATION)
        elif operation.kind == "MERGING":
            self._set_stage(unit, UnitStage.WAITING_S4)
        elif operation.kind == "PRE_BRAZE_INSPECTION":
            self.faults.complete_recovery(unit.unit_id, self.sim_time)
            if unit.stage is UnitStage.FINS_REVIEW:
                self.camera_coordination.complete(
                    unit.unit_id,
                    operation.kind,
                    now=self.sim_time,
                    result="PASS",
                )
                self._set_stage(unit, UnitStage.WAITING_MERGE)
            else:
                self._set_stage(unit, UnitStage.WAITING_BUFFER)
        elif operation.kind == "POST_BRAZE_INSPECTION":
            self._set_stage(unit, UnitStage.WAITING_OUTPUT)
        elif operation.kind == "OUTPUT_GATE_OPEN":
            self._output_gate_ready = True
            self._event("OUTPUT_GATE_OPENED", unit_id=unit.unit_id)
        elif operation.kind == "OUTPUT_DELIVERY":
            assert unit.tray_id is not None
            self.flow.mark_product_removed(unit.tray_id, now=self.sim_time)
            self._set_stage(unit, UnitStage.PRODUCT_REMOVED)
            self._output_gate_open = False
            self._output_gate_ready = False
            self._start(
                "OUTPUT_GATE",
                unit,
                "OUTPUT_GATE_CLOSE",
                self.durations.furnace_door,
            )
        elif operation.kind == "OUTPUT_GATE_CLOSE":
            assert unit.tray_id is not None
            self._event("OUTPUT_GATE_CLOSED", unit_id=unit.unit_id)
            self.flow.start_virtual_return(unit.tray_id, now=self.sim_time)
            self._set_stage(unit, UnitStage.VIRTUAL_RETURN)
        elif operation.kind == "VIRTUAL_RETURN":
            assert unit.tray_id is not None
            self.flow.complete_virtual_return(unit.tray_id, now=self.sim_time)
            self._set_stage(unit, UnitStage.COMPLETE)
            unit.completed_at = self.sim_time
            self._event("UNIT_COMPLETED", unit_id=unit.unit_id, order_id=unit.order_id)
        elif operation.kind == "FURNACE_FRONT_OPEN":
            self._event("FURNACE_FRONT_DOOR_OPENED", unit_id=unit.unit_id)
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
            self._event("FURNACE_REAR_DOOR_OPENED", unit_id=unit.unit_id)
        elif operation.kind == "FURNACE_UNLOAD_TRAY":
            self._set_stage(unit, UnitStage.POST_BRAZE_INSPECTION)
            self._start(
                "POST_CAMERA",
                unit,
                "POST_BRAZE_INSPECTION",
                self.durations.post_braze_inspection,
            )
            if not any(layer.tray_id is not None for layer in self.furnace.state.layers):
                self.furnace.close_rear(now=self.sim_time)
                self._start(
                    "FURNACE_DOOR",
                    unit,
                    "FURNACE_REAR_CLOSE",
                    self.durations.furnace_door,
                )
        elif operation.kind == "FURNACE_REAR_CLOSE":
            self._event("FURNACE_REAR_DOOR_CLOSED")
            if not self._batch_recorded:
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
            if operation.manual_hold_fault_ids:
                operation.manual_hold_fault_ids = tuple(
                    fault_id
                    for fault_id in operation.manual_hold_fault_ids
                    if not self.faults.faults[fault_id].recovered
                )
                if operation.manual_hold_fault_ids:
                    continue
            # An isolated arm is physically stopped, so its logical operation
            # clock must stop as well.  Otherwise the task graph can finish an
            # action while the MuJoCo arm is visibly frozen mid-path.
            if not self._resource_online(operation.resource):
                continue
            advance = dt
            if operation.fault_hold_remaining_s > 0.0:
                held = min(advance, operation.fault_hold_remaining_s)
                operation.fault_hold_remaining_s = max(
                    0.0,
                    operation.fault_hold_remaining_s - held,
                )
                advance -= held
            operation.remaining_s = max(0.0, operation.remaining_s - advance)
            if operation.remaining_s <= 1.0e-9 and self._operation_can_complete(operation):
                completed.append(operation)
        for operation in completed:
            self.operations.pop(operation.resource, None)
            recovered_faults = self.faults.complete_bound_operation_recovery(
                operation.hold_fault_ids,
                resource=operation.resource,
                unit_id=operation.unit_id,
                operation_kind=operation.kind,
                now=self.sim_time,
            )
            for fault_id in recovered_faults:
                self._event(
                    "RECOVERY_RETRY_COMPLETED",
                    fault_id=fault_id,
                    resource=operation.resource,
                    unit_id=operation.unit_id,
                    kind=operation.kind,
                )
            detected = self.faults.detect_for_operation(
                operation,
                now=self.sim_time,
                unit_lookup=self.units,
            )
            if detected:
                self._event(
                    "OPERATION_CANCELLED",
                    resource=operation.resource,
                    unit_id=operation.unit_id,
                    kind=operation.kind,
                    reason="QUALITY_DEFECT_DETECTED",
                )
                for record in detected:
                    self._event(
                        "FAULT_DETECTED",
                        fault_id=record.fault_id,
                        fault_type=record.fault_type.value,
                        source=record.source,
                        unit_id=record.details.get("unit_id"),
                        target=record.details.get("target"),
                    )
                    self._event(
                        "FAULT_INJECTED",
                        fault_id=record.fault_id,
                        fault_type=record.fault_type.value,
                        source=record.source,
                        unit_id=record.details.get("unit_id"),
                        recoverable=record.recoverable,
                    )
                self._apply_rollbacks()
                continue
            self._complete_operation(operation)

    def _admit_next_unit(self) -> bool:
        if not self._owner_free(TrayOwner.S1) or not self._target_available(TrayOwner.S1):
            return False
        queued = self._waiting_units(UnitStage.QUEUED)
        admission_callback = (
            None
            if self._execution_gate is None
            else getattr(self._execution_gate, "unit_admission_allowed", None)
        )
        if admission_callback is not None:
            queued = [unit for unit in queued if admission_callback(unit.unit_id)]
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
                not self._s2a_inspection_clearance_required()
                and self._tray_ready(unit)
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
        for unit in self._waiting_units(UnitStage.WAITING_BRAZING_REVIEW):
            if (
                unit.branch is InstallBranch.ARM3_B
                and self._install_transfer_allowed(unit)
                and self._tray_ready(unit)
                and self._owner_free(TrayOwner.INSTALL_B)
                and self._target_available(TrayOwner.INSTALL_B)
            ):
                self._handoff(
                    unit,
                    TrayOwner.S2B,
                    TrayOwner.INSTALL_B,
                    TrayPhase.FIN_INSTALLATION,
                )
                self._set_stage(unit, UnitStage.BRAZING_REVIEW)
                changed = True
                break
        for unit in self._waiting_units(UnitStage.WAITING_INSTALL):
            self._reroute_blocked_install(unit)
            if (
                self._install_transfer_allowed(unit)
                and unit.branch is InstallBranch.ARM3_B
                and unit.tray_id is not None
                and self._tray_ready(unit)
                and self.flow.get(unit.tray_id).owner is TrayOwner.INSTALL_B
            ):
                self._set_stage(unit, UnitStage.FIN_INSTALLATION)
                changed = True
                break
            if (
                self._tray_ready(unit)
                and self._install_transfer_allowed(unit)
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
                and self._install_transfer_allowed(unit)
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
        for unit in self._waiting_units(UnitStage.WAITING_FINS_REVIEW):
            if unit.tray_id is None or not self._tray_ready(unit):
                continue
            if self.flow.get(unit.tray_id).owner is TrayOwner.INSTALL_B:
                self._set_stage(unit, UnitStage.FINS_REVIEW)
                changed = True
                break
        for unit in self._waiting_units(UnitStage.WAITING_MERGE):
            assert unit.tray_id is not None
            if not self._tray_ready(unit):
                continue
            owner = self.flow.get(unit.tray_id).owner
            if owner is TrayOwner.S2B and unit.branch is InstallBranch.ARM3_B:
                # A completed S3B pallet can temporarily yield to S2B when an
                # S4 fin correction needs Arm3's station.  Keep its completed
                # fin state and WAITING_MERGE stage intact; once every active
                # FIN_REINSTALL plan has passed reinspection, return it along
                # the same branch rail without replaying any process task.
                recovery_claims_s3b = any(
                    plan.strategy == "FIN_REINSTALL"
                    and plan.status is RecoveryStatus.RUNNING
                    for plan in self.faults.plans.values()
                )
                if recovery_claims_s3b:
                    continue
                if self._owner_free(TrayOwner.INSTALL_B) and self._target_available(
                    TrayOwner.INSTALL_B
                ):
                    self._handoff(
                        unit,
                        TrayOwner.S2B,
                        TrayOwner.INSTALL_B,
                        TrayPhase.MERGE_WAIT,
                    )
                    self._event(
                        "RECOVERY_STATION_YIELD_COMPLETED",
                        unit_id=unit.unit_id,
                        tray_id=unit.tray_id,
                        source=TrayOwner.S2B.value,
                        target=TrayOwner.INSTALL_B.value,
                    )
                    changed = True
                    break
                continue
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
            if not self._output_gate_ready:
                continue
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
        if "POST_CAMERA" not in self.operations and self._resource_online("POST_CAMERA"):
            waiting_post = [
                unit
                for unit in self._waiting_units(UnitStage.POST_BRAZE_INSPECTION)
                if self._tray_ready(unit)
            ]
            if waiting_post:
                changed |= self._start(
                    "POST_CAMERA",
                    waiting_post[0],
                    "POST_BRAZE_INSPECTION",
                    self.durations.post_braze_inspection,
                )
        if "ARM2" not in self.operations and self._resource_online("ARM2"):
            waiting = [unit for unit in self._waiting_units(UnitStage.DISPENSING) if self._tray_ready(unit)]
            if waiting:
                changed |= self._start("ARM2", waiting[0], "DISPENSING", self.durations.dispensing)
        # Detection always gets Arm3 before another fin is picked. An active
        # INSTALL_FIN operation remains non-preemptible until this tick ends.
        if "ARM3" not in self.operations and self._resource_online("ARM3"):
            inspection_candidates: list[tuple[V2UnitState, str, float]] = []
            for unit in self._waiting_units(UnitStage.BRAZING_REVIEW, UnitStage.FINS_REVIEW):
                if not self._tray_ready(unit):
                    continue
                kind = (
                    "MATERIAL_INSPECTION"
                    if unit.stage is UnitStage.BRAZING_REVIEW
                    else "PRE_BRAZE_INSPECTION"
                )
                duration = (
                    self.durations.material_inspection
                    if kind == "MATERIAL_INSPECTION"
                    else self.durations.pre_braze_inspection
                )
                inspection_candidates.append((unit, kind, duration))
            inspection_candidates.extend(
                (unit, "PRE_BRAZE_INSPECTION", self.durations.pre_braze_inspection)
                for unit in self._waiting_units(UnitStage.PRE_BRAZE_INSPECTION)
                if self._tray_ready(unit)
            )
            inspection_candidates.extend(
                (unit, "MATERIAL_INSPECTION", self.durations.material_inspection)
                for unit in self._waiting_units(UnitStage.MATERIAL_INSPECTION)
                if self._tray_ready(unit)
            )
            for unit, kind, duration in inspection_candidates:
                if self._start("ARM3", unit, kind, duration):
                    changed = True
                    break
        if "MERGE" not in self.operations and self._resource_online("MERGE"):
            waiting = [unit for unit in self._waiting_units(UnitStage.MERGING) if self._tray_ready(unit)]
            if waiting:
                changed |= self._start("MERGE", waiting[0], "MERGING", self.durations.merge)
        # Arm1 prioritises a tray already committed to installation; only then
        # does it admit another base plate.
        if "ARM1" not in self.operations and self._resource_online("ARM1"):
            install = [
                unit
                for unit in self._waiting_units(UnitStage.FIN_INSTALLATION)
                if unit.branch is InstallBranch.ARM1_A
                and (unit.fins_installed < unit.fin_count or unit.rework_fin_index is not None)
                and self._tray_ready(unit)
            ]
            install_started = bool(
                install and self._start("ARM1", install[0], "INSTALL_FIN", self.durations.arm1_fin)
            )
            changed |= install_started
            if not install_started:
                # A physically ready fin tray is only a candidate.  The
                # unified scheduler may intentionally withhold its permit to
                # keep the vacuum tool resident and stage the next base.  Do
                # not let that denied candidate suppress S1 admission and
                # leave Arm1 idle in a circular wait.
                base_loading = [
                    unit for unit in self._waiting_units(UnitStage.BASE_LOADING) if self._tray_ready(unit)
                ]
                if base_loading:
                    changed |= self._start(
                        "ARM1",
                        base_loading[0],
                        "BASE_LOADING",
                        self.durations.base_load,
                    )
        if "ARM3" not in self.operations and self._resource_online("ARM3"):
            install = [
                unit
                for unit in self._waiting_units(UnitStage.FIN_INSTALLATION)
                if unit.branch is InstallBranch.ARM3_B
                and (unit.fins_installed < unit.fin_count or unit.rework_fin_index is not None)
                and self._tray_ready(unit)
            ]
            if install and self._arm3_fin_fits_before_inspection(install[0]):
                changed |= self._start("ARM3", install[0], "INSTALL_FIN", self.durations.arm3_fin)
        if "OUTPUT" not in self.operations and self._resource_online("OUTPUT"):
            delivering = [
                unit for unit in self._waiting_units(UnitStage.DELIVERING) if self._tray_ready(unit)
            ]
            if delivering:
                changed |= self._start(
                    "OUTPUT",
                    delivering[0],
                    "OUTPUT_DELIVERY",
                    self.durations.output_delivery,
                )
        if (
            "OUTPUT_GATE" not in self.operations
            and self._resource_online("OUTPUT_GATE")
            and not self._output_gate_open
        ):
            waiting_output = [
                unit
                for unit in self._waiting_units(UnitStage.WAITING_OUTPUT)
                if self._tray_ready(unit)
                and self._owner_free(TrayOwner.OUTPUT)
                and self._target_available(TrayOwner.OUTPUT)
            ]
            if waiting_output:
                started = self._start(
                    "OUTPUT_GATE",
                    waiting_output[0],
                    "OUTPUT_GATE_OPEN",
                    self.durations.furnace_door,
                )
                if started:
                    self._output_gate_open = True
                    self._output_gate_ready = False
                    changed = True
        for unit in self._waiting_units(UnitStage.VIRTUAL_RETURN):
            resource = f"RETURN:{unit.unit_id}"
            if resource not in self.operations and self._resource_online(resource):
                changed |= self._start(
                    resource,
                    unit,
                    "VIRTUAL_RETURN",
                    self.durations.virtual_return,
                )
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
            batch_recipe = self.units[self._active_batch_units[0]].batch_recipe
            compatible = [unit for unit in buffered if batch_recipe.compatible_with(unit.batch_recipe)]
            if not compatible:
                return False
            unit = compatible[0]
            assert unit.tray_id is not None
            self.furnace.append_loading_tray(
                unit.tray_id,
                unit.batch_recipe,
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
        if not self._operation_start_allowed(
            "FURNACE_DOOR",
            selected[0],
            "FURNACE_FRONT_OPEN",
        ):
            return False
        self.furnace.plan_batch(
            tuple((unit.tray_id or unit.unit_id, unit.batch_recipe) for unit in selected),
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
            available_layers = sorted(
                (
                    layer.index
                    for layer in self.furnace.state.layers
                    if layer.tray_id is None and self.faults.layer_available(layer.index)
                ),
                reverse=True,
            )
            if not available_layers:
                return False
            layer = available_layers[0]
            if not self._operation_start_allowed(
                "FURNACE_TRANSFER",
                unit,
                "FURNACE_LOAD_TRAY",
            ):
                return False
            unit.furnace_layer = layer
            for fault_layer in sorted(self.faults.unavailable_rack_layers):
                if self.faults.mark_rack_reallocated(
                    fault_layer,
                    layer,
                    unit.unit_id,
                    self.sim_time,
                ):
                    self._event(
                        "RACK_LAYER_REALLOCATED",
                        unit_id=unit.unit_id,
                        fault_layer=fault_layer,
                        selected_layer=layer,
                    )
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
        if (
            not self.furnace.state.rear_door_open
            and "FURNACE_DOOR" not in self.operations
            and self._resource_online("FURNACE_DOOR")
        ):
            leader = self.units[self._active_batch_units[-1]]
            if not self._operation_start_allowed(
                "FURNACE_DOOR",
                leader,
                "FURNACE_REAR_OPEN",
            ):
                return False
            self.furnace.open_rear(now=self.sim_time)
            self._start(
                "FURNACE_DOOR",
                leader,
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
        if not self._operation_start_allowed(
            "FURNACE_TRANSFER",
            unit,
            "FURNACE_UNLOAD_TRAY",
        ):
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
            # Empty-tray admission is logistics work, not an Arm1 operation.
            # Stage the next released unit at S1 as soon as the station and
            # route are clear, even while Arm1 is finishing a non-preemptible
            # fin installation elsewhere.  Arm1 still cannot touch the tray
            # until the physical S1 readiness gate passes.
            changed |= self._admit_next_unit()
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
        # Faults are serviced before operations advance: an armed request fires
        # against the operation that is running *now*, and an isolated resource
        # must already be excluded from this tick's dispatch.
        self._service_faults()
        if not self.faults.cell_available():
            return self.snapshot()
        self._advance_operations(dt)
        self.furnace.update(self.sim_time)
        self._arm3_reservation_blocked = False
        self._dispatch()
        if self._arm3_reservation_blocked:
            self.arm3_inspection_reservation_wait_s += dt
        prepositioning = self.prepositioning_snapshot()
        if prepositioning:
            self.robot_transport_overlap_s += dt
            for resource in prepositioning:
                self.preposition_seconds[resource] += dt
        physically_occupied = {
            tray.owner
            for tray in self.flow.trays
            if tray.owner in {TrayOwner.S1, TrayOwner.S2A}
            and self._execution_gate is not None
            and self._execution_gate.tray_ready(tray.tray_id, tray.owner)
        }
        if {TrayOwner.S1, TrayOwner.S2A} <= physically_occupied:
            self.s1_s2a_dual_occupancy_s += dt
        active_wip = sum(tray.owner is not TrayOwner.EMPTY_BUFFER for tray in self.flow.trays)
        self.maximum_wip = max(self.maximum_wip, active_wip)
        return self.snapshot()

    # ------------------------------------------------------------------ faults
    def _service_faults(self) -> None:
        """Fire armed requests, apply rollbacks and release auto-recoveries."""

        for plan in self.faults.service_manual_reviews(self.sim_time):
            record = self.faults.faults[plan.fault_id]
            if record.fault_type.value == "ARM_UNAVAILABLE":
                self._event(
                    "RESOURCE_RECOVERED",
                    resource=str(record.details.get("target") or record.source).upper(),
                )
            elif record.fault_type.value == "FURNACE_PROFILE" and plan.unit_id:
                unit = self.units.get(plan.unit_id)
                if unit is not None and unit.stage is UnitStage.POST_BRAZE_INSPECTION:
                    self._set_stage(unit, UnitStage.WAITING_OUTPUT)
            self._event(
                "MANUAL_REVIEW_COMPLETED",
                recovery_id=plan.recovery_id,
                fault_id=record.fault_id,
                fault_type=record.fault_type.value,
                message=plan.message,
            )
        for resource in self.faults.service_auto_recovery(self.sim_time):
            self._event("RESOURCE_RECOVERED", resource=resource)
        manifested = self.faults.manifest_matching(
            self.operations.values(),
            now=self.sim_time,
            unit_lookup=self.units,
        )
        for defect in manifested:
            self._event(
                "FAULT_MANIFESTED",
                defect_id=defect.defect_id,
                fault_type=defect.fault_type.value,
                visual_type=defect.visual_type,
                unit_id=defect.unit_id,
                target=defect.target,
                operation=defect.source_operation,
            )
        fired = self.faults.fire_matching(
            self.operations.values(),
            now=self.sim_time,
            unit_lookup=self.units,
        )
        for record in fired:
            self._event(
                "FAULT_INJECTED",
                fault_id=record.fault_id,
                fault_type=record.fault_type.value,
                source=record.source,
                unit_id=record.details.get("unit_id"),
                recoverable=record.recoverable,
            )
            unit_id = str(record.details.get("unit_id") or "")
            requested_hold = float(record.details.get("duration_s") or 4.0)
            hold = self.faults.take_hold(unit_id, requested_hold) if unit_id else 0.0
            if hold <= 0.0:
                continue
            operation = next(
                (
                    item
                    for item in self.operations.values()
                    if item.unit_id == unit_id and item.resource == record.source
                ),
                None,
            )
            if operation is None:
                # Defensive fallback: preserve the owed delay so a committed
                # mechanism transition can consume it at its next safe start.
                self.faults.unit_holds[unit_id] = self.faults.unit_holds.get(unit_id, 0.0) + hold
                continue
            plan = self.faults.plans.get(record.recovery_id or "")
            if plan is not None and plan.status is RecoveryStatus.MANUAL_REVIEW:
                operation.manual_hold_fault_ids = (
                    *operation.manual_hold_fault_ids,
                    record.fault_id,
                )
                self.faults.bind_hold_to_operation(
                    record.fault_id,
                    resource=operation.resource,
                    operation_kind=operation.kind,
                    seconds=hold,
                )
                self._event(
                    "FAULT_HOLD_APPLIED",
                    resource=operation.resource,
                    unit_id=unit_id,
                    kind=operation.kind,
                    seconds=hold,
                    applied_to="CURRENT_OPERATION",
                    manual_confirmation=True,
                    fault_id=record.fault_id,
                )
                continue
            # ``fault_hold_remaining_s`` is a separate frozen-clock interval.
            # Do not also add it to ``remaining_s`` or the same timeout would
            # be charged twice (once while frozen and once after motion resumes).
            operation.fault_hold_remaining_s += hold
            operation.hold_fault_ids = (*operation.hold_fault_ids, record.fault_id)
            self.faults.bind_hold_to_operation(
                record.fault_id,
                resource=operation.resource,
                operation_kind=operation.kind,
                seconds=hold,
            )
            self._event(
                "FAULT_HOLD_APPLIED",
                resource=operation.resource,
                unit_id=unit_id,
                kind=operation.kind,
                seconds=hold,
                applied_to="CURRENT_OPERATION",
                fault_id=record.fault_id,
            )
        self._apply_rollbacks()

    def _apply_rollbacks(self) -> None:
        """Roll a failed unit back to the stage that must be redone.

        V1 achieves recovery by inserting rework tasks into a DAG.  V2 has no
        DAG, so the equivalent is returning the unit's stage machine to the point
        before the failed operation and letting the dispatcher run it again.

        The rollback is applied exactly once per plan.  Note that the unit is
        often *already* sitting in the rollback stage (a braze defect is detected
        during ``MATERIAL_INSPECTION``, whose stage is still ``DISPENSING``);
        skipping those would silently drop the rework, so completion is keyed on
        an explicit ``rollback_applied`` flag rather than on a stage difference.
        """

        for plan in list(self.faults.plans.values()):
            if plan.status is not RecoveryStatus.RUNNING or plan.rollback_applied:
                continue
            if not plan.unit_id or not plan.rollback_stage:
                continue
            unit = self.units.get(plan.unit_id)
            if unit is None:
                continue
            try:
                stage = UnitStage(plan.rollback_stage)
            except ValueError:
                continue
            actual_stage = stage
            if unit.tray_id is not None:
                owner = self.flow.get(unit.tray_id).owner
                if plan.strategy == "LOCAL_BRAZING_REWORK" and owner is TrayOwner.S2B:
                    # A camera can only report a defect after the pallet has
                    # settled at S2B.  Keep this physical gate here as well so
                    # an externally driven/manual detection cannot change the
                    # logical owner while the carrier is still moving.
                    if not self._tray_ready(unit):
                        continue
                    if not (self._owner_free(TrayOwner.S2A) and self._target_available(TrayOwner.S2A)):
                        continue
                    self._handoff(
                        unit,
                        TrayOwner.S2B,
                        TrayOwner.S2A,
                        TrayPhase.DISPENSING,
                    )
                    actual_stage = UnitStage.DISPENSING
                    self._event(
                        "RECOVERY_RETURN_STARTED",
                        unit_id=unit.unit_id,
                        source=TrayOwner.S2B.value,
                        target=TrayOwner.S2A.value,
                        strategy=plan.strategy,
                    )
                elif plan.strategy == "LOCAL_BRAZING_REWORK":
                    # The logical stage must never rewind without the carrier
                    # handoff.  Keep the plan pending until S2B is the actual
                    # owner again; normal dispatch is filtered above while it
                    # waits for this reservation.
                    continue
                elif plan.strategy == "FIN_REINSTALL" and owner is TrayOwner.S4:
                    # Quality reseating is an Arm3 camera+gripper skill.  Keep
                    # the defective assembly intact at S4 until S3B is free,
                    # then return the same pallet to that dedicated correction
                    # station.  Reusing the original branch could send the job
                    # to Arm1 and replay the normal magazine-pick sequence.
                    target_branch = InstallBranch.ARM3_B
                    target_owner = TrayOwner.INSTALL_B
                    if not self._tray_ready(unit):
                        continue
                    if not self._owner_free(target_owner):
                        blocker_tray = next(
                            (
                                tray
                                for tray in self.flow.trays
                                if tray.owner is target_owner and tray.unit_id != unit.unit_id
                            ),
                            None,
                        )
                        blocker = (
                            None
                            if blocker_tray is None or blocker_tray.unit_id is None
                            else self.units.get(blocker_tray.unit_id)
                        )
                        if (
                            blocker is not None
                            and blocker.stage is UnitStage.WAITING_MERGE
                            and blocker.branch is InstallBranch.ARM3_B
                            and blocker.fins_installed >= blocker.fin_count
                            and blocker.rework_fin_index is None
                            and self._tray_ready(blocker)
                            and self._owner_free(TrayOwner.S2B)
                            and self._target_available(TrayOwner.S2B)
                        ):
                            self._handoff(
                                blocker,
                                TrayOwner.INSTALL_B,
                                TrayOwner.S2B,
                                TrayPhase.MERGE_WAIT,
                            )
                            self._event(
                                "RECOVERY_STATION_YIELD_STARTED",
                                unit_id=blocker.unit_id,
                                tray_id=blocker.tray_id,
                                recovery_unit_id=unit.unit_id,
                                source=TrayOwner.INSTALL_B.value,
                                target=TrayOwner.S2B.value,
                                reason_zh="S4翅片纠偏需要Arm3工位，已完工托盘沿原滑轨临时让路",
                            )
                        continue
                    if not (self._owner_free(target_owner) and self._target_available(target_owner)):
                        continue
                    previous_branch = unit.branch
                    if previous_branch is not target_branch:
                        if previous_branch is not None:
                            self.install_branch_counts[previous_branch] = max(
                                0,
                                self.install_branch_counts[previous_branch] - 1,
                            )
                        self.install_branch_counts[target_branch] += 1
                        unit.branch = target_branch
                        self._event(
                            "INSTALL_REASSIGNED",
                            unit_id=unit.unit_id,
                            tray_id=unit.tray_id,
                            previous_branch=(None if previous_branch is None else previous_branch.value),
                            branch=target_branch.value,
                            reason_zh="焊前视觉检出翅片偏位，转入S3B由Arm3原槽位纠偏",
                        )
                    self._handoff(
                        unit,
                        TrayOwner.S4,
                        target_owner,
                        TrayPhase.FIN_INSTALLATION,
                    )
                    actual_stage = UnitStage.FIN_INSTALLATION
                    self._event(
                        "RECOVERY_RETURN_STARTED",
                        unit_id=unit.unit_id,
                        source=TrayOwner.S4.value,
                        target=target_owner.value,
                        strategy=plan.strategy,
                    )
                elif plan.strategy == "FIN_REINSTALL":
                    # As above, do not turn a failed S4 inspection into a
                    # state-only rewind when the S3B return corridor is busy.
                    continue
            # Cancel any in-flight operation for this unit before rewinding, so a
            # completing operation cannot advance a unit that is being redone.
            for resource, operation in list(self.operations.items()):
                if operation.unit_id == plan.unit_id:
                    del self.operations[resource]
                    self._event(
                        "OPERATION_CANCELLED",
                        resource=resource,
                        unit_id=plan.unit_id,
                        kind=operation.kind,
                        reason="RECOVERY_ROLLBACK",
                    )
            rollback_fin_count = unit.fins_installed
            if actual_stage is UnitStage.FIN_INSTALLATION:
                # Keep every already installed fin physically present.  The
                # exact failed index is carried separately so the repair actor
                # returns to that slot instead of pretending the last fin failed.
                defect = next(
                    (item for item in self.faults.physical_faults.values() if item.fault_id == plan.fault_id),
                    None,
                )
                unit.rework_fin_index = (None if defect is None else defect.operation_index) or max(
                    1, unit.fins_installed
                )
                rollback_fin_count = max(0, unit.fins_installed - 1)
            plan.rollback_applied = True
            # Rework costs more than the interrupted operation's remainder: the
            # defect itself must be repaired.  Charging it here keeps recovery
            # visible in makespan rather than looking free.
            defect = next(
                (item for item in self.faults.physical_faults.values() if item.fault_id == plan.fault_id),
                None,
            )
            self._rework_effort[plan.unit_id] = _RecoveryWork(
                effort_factor=plan.effort_factor,
                strategy=plan.strategy,
                fault_type="" if defect is None else defect.fault_type.value,
                target_index=None if defect is None else defect.operation_index,
            )
            self._set_stage(unit, actual_stage)
            self._event(
                "RECOVERY_ROLLBACK",
                unit_id=plan.unit_id,
                recovery_id=plan.recovery_id,
                stage=stage.value,
                actual_stage=actual_stage.value,
                effort_factor=plan.effort_factor,
                fins_installed=rollback_fin_count,
                fin_index=unit.rework_fin_index,
            )

    def inject_fault(
        self,
        fault_type: str,
        *,
        target: str = "",
        severity: str = "recoverable",
        auto_recover: bool = True,
        duration_s: float | None = None,
        label_zh: str = "",
    ) -> dict[str, Any]:
        """Arm a manual fault; returns the request snapshot for the UI."""

        from ..fault_catalog import MANUAL_FAULT_CATALOG

        requested_type = str(fault_type).strip().upper()
        definition = MANUAL_FAULT_CATALOG.get(requested_type)
        runtime_type = requested_type if definition is None else definition.runtime_fault
        if not runtime_type:
            raise ValueError(f"V2 暂不支持故障 {requested_type} 的运行时执行")
        request = self.faults.arm(
            runtime_type,
            target=target,
            severity=severity,
            auto_recover=auto_recover,
            duration_s=duration_s,
            label_zh=label_zh or (requested_type if definition is None else definition.label_zh),
            visual_type=requested_type,
            now=self.sim_time,
        )
        self._event(
            "FAULT_ARMED",
            request_id=request.request_id,
            fault_type=request.fault_type.value,
            visual_type=request.visual_type,
            target=request.target,
            status=request.status,
        )
        return request.as_dict()

    def recover_resource(self, resource_id: str) -> bool:
        recovered = self.faults.recover_resource(resource_id, self.sim_time)
        if recovered:
            self._event("RESOURCE_RECOVERED", resource=str(resource_id).upper())
        return recovered

    def recovery_action(self, recovery_id: str, action: str) -> bool:
        applied = self.faults.action(recovery_id, action, self.sim_time)
        if applied:
            self._event("RECOVERY_ACTION", recovery_id=recovery_id, action=action)
        return applied

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
        self._output_gate_open = False
        self._output_gate_ready = False
        self.completed_batches.clear()
        self.install_branch_counts = {branch: 0 for branch in InstallBranch}
        self.scheduled_parallel_install_seconds = 0.0
        self.upstream_work_during_brazing_s = 0.0
        self.robot_transport_overlap_s = 0.0
        self.s1_s2a_dual_occupancy_s = 0.0
        self.preposition_seconds = {resource: 0.0 for resource in ("ARM1", "ARM2", "ARM3")}
        self.arm3_inspection_reservation_wait_s = 0.0
        self._arm3_reservation_blocked = False
        self.maximum_wip = 0
        self.dispatcher.reset()
        self.faults.reset()
        self.camera_coordination.reset()
        self._rework_effort.clear()

    def capture_digital_twin(self, *, emit_event: bool = False) -> DigitalTwinSnapshot:
        """Capture an immutable shadow view without changing V2 execution."""

        state = self.snapshot()
        snapshot = DigitalTwinSnapshot.from_mapping(
            state,
            source_name="DualLineRuntime",
            captured_at=float(state.get("sim_time", 0.0)),
            plan_version=len(self.events),
        )
        if emit_event:
            encoded = DecisionEvent(
                event_type=EventType.STATE_SNAPSHOT_CAPTURED,
                sim_time=snapshot.sim_time,
                source="DualLineRuntime",
                plan_version=snapshot.plan_version,
                trigger="EXPLICIT_CAPTURE",
                payload={"fingerprint": snapshot.fingerprint},
            ).as_dict()
            encoded["type"] = encoded["event_type"]
            self.events.append(encoded)
        return snapshot

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
                    "duration_s": operation.duration_s,
                    "started_at": operation.started_at,
                    "progress": (
                        1.0
                        if operation.duration_s <= 0.0
                        else max(
                            0.0,
                            min(1.0, 1.0 - operation.remaining_s / operation.duration_s),
                        )
                    ),
                    "recovery": operation.recovery,
                    "recovery_strategy": operation.recovery_strategy,
                    "recovery_fault_type": operation.recovery_fault_type,
                    "recovery_target_index": operation.recovery_target_index,
                }
                for resource, operation in sorted(self.operations.items())
            },
            "prepositioning": self.prepositioning_snapshot(),
            "arm3_inspection_windows": [window.as_dict() for window in self.arm3_inspection_windows()],
            "rolling_horizon_scheduler": self.dispatcher.snapshot(),
            "install_branch_counts": {
                branch.value: count for branch, count in self.install_branch_counts.items() if count > 0
            },
            "scheduled_parallel_install_seconds": round(self.scheduled_parallel_install_seconds, 6),
            "furnace": {
                **self.furnace.as_dict(),
                "completed_batches": len(self.completed_batches),
                "last_batch": last_batch,
            },
            "output": {
                "gate_open": self._output_gate_open,
                "gate_ready": self._output_gate_ready,
            },
            "camera_coordination": self.camera_coordination_snapshot(),
            "arm3_camera_plan": self.camera_coordination_snapshot(),
            "metrics": {
                "upstream_work_during_brazing_s": round(self.upstream_work_during_brazing_s, 6),
                "robot_transport_overlap_s": round(self.robot_transport_overlap_s, 6),
                "s1_s2a_dual_occupancy_s": round(self.s1_s2a_dual_occupancy_s, 6),
                "arm3_inspection_reservation_wait_s": round(
                    self.arm3_inspection_reservation_wait_s,
                    6,
                ),
                "preposition_seconds": {
                    resource: round(seconds, 6) for resource, seconds in self.preposition_seconds.items()
                },
                "maximum_wip": self.maximum_wip,
                "fault_count": len(self.faults.faults),
                "recovered_fault_count": sum(1 for record in self.faults.faults.values() if record.recovered),
                "recovery_rate": (
                    0.0
                    if not self.faults.faults
                    else sum(1 for r in self.faults.faults.values() if r.recovered) / len(self.faults.faults)
                ),
            },
            **self.faults.snapshot(),
            "events": list(self.events),
        }


__all__ = ["DualLineRuntime", "UnitStage", "V2OrderState", "V2UnitState"]
