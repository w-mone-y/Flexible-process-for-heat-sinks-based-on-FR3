"""V2 fault injection, recovery planning and resource isolation.

Ported from V1's ``manufacturing_runtime`` + ``recovery`` pair, but rebuilt on
V2's substrate.  Three things carry over unchanged because they earned it:

*   **The catalogue** (:mod:`brazing_sim.fault_catalog`) — operator fault codes with
    Chinese labels and payload validation, entirely scene-independent.
*   **The fault/recovery data model** (:mod:`brazing_sim.recovery.fault_models`).
*   **Deferred arming.**  V1's best idea: an operator injects a fault *before*
    the relevant operation runs, the request sits ``ARMED``, and it fires when a
    matching operation actually starts.  The operator never has to time it.

What had to be rewritten: V1 plans recovery by *graph surgery* — inserting rework
tasks into a ``TaskGraph`` and re-pointing edges.  V2 has no DAG; it advances a
per-unit ``UnitStage`` machine.  Recovery here therefore works by **stage
rollback**: a failed unit returns to the stage that must be redone, its rework
counter increments, and the normal dispatcher picks it up again.  The strategy
table, the per-target retry ceiling and the pause/resume/retry/manual verbs are
preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..recovery.fault_models import FaultRecord, FaultType, RecoveryStatus
from ..fault_catalog import MANUAL_FAULT_CATALOG

# Quality faults use two different triggers.  They first *manifest* while the
# work is being produced and remain latent/visible on the pallet.  A later
# camera operation detects them and only then creates a FaultRecord/recovery.
# Equipment faults keep the immediate behaviour because there is no latent
# workpiece defect to inspect.
FAULT_MANIFEST_KINDS: dict[FaultType, frozenset[str]] = {
    FaultType.BRAZING_MISSING: frozenset({"DISPENSING"}),
    FaultType.BRAZING_PATH_DEVIATION: frozenset({"DISPENSING"}),
    FaultType.FIN_PICK_FAILED: frozenset({"INSTALL_FIN"}),
    FaultType.FIN_GEOMETRY_FAILED: frozenset({"INSTALL_FIN"}),
    FaultType.FURNACE_PROFILE: frozenset({"FURNACE_FRONT_CLOSE"}),
}

FAULT_DETECTION_KINDS: dict[FaultType, frozenset[str]] = {
    FaultType.BRAZING_MISSING: frozenset({"MATERIAL_INSPECTION"}),
    FaultType.BRAZING_PATH_DEVIATION: frozenset({"MATERIAL_INSPECTION"}),
    FaultType.FIN_PICK_FAILED: frozenset({"PRE_BRAZE_INSPECTION"}),
    FaultType.FIN_GEOMETRY_FAILED: frozenset({"PRE_BRAZE_INSPECTION"}),
    FaultType.FURNACE_PROFILE: frozenset({"POST_BRAZE_INSPECTION"}),
}

# Public compatibility view used by API/tests: every deferred fault has a real
# operation trigger.  It is intentionally the union, not the detection-only
# table that caused defects to appear too late.
FAULT_OPERATION_KINDS: dict[FaultType, frozenset[str]] = {
    **{
        fault_type: FAULT_MANIFEST_KINDS.get(fault_type, frozenset())
        | FAULT_DETECTION_KINDS.get(fault_type, frozenset())
        for fault_type in set(FAULT_MANIFEST_KINDS) | set(FAULT_DETECTION_KINDS)
    },
    FaultType.ELEVATOR_TIMEOUT: frozenset({"FURNACE_LOAD_TRAY", "FURNACE_UNLOAD_TRAY"}),
    FaultType.FORK_TIMEOUT: frozenset({"FURNACE_LOAD_TRAY", "FURNACE_UNLOAD_TRAY"}),
    FaultType.FURNACE_DOOR_INTERLOCK: frozenset(
        {
            "FURNACE_FRONT_OPEN",
            "FURNACE_FRONT_CLOSE",
            "FURNACE_REAR_OPEN",
            "FURNACE_REAR_CLOSE",
        }
    ),
    # Equipment and safety faults are not tied to one process step.
    FaultType.ARM_UNAVAILABLE: frozenset(),
    FaultType.RACK_LAYER_UNAVAILABLE: frozenset(),
    FaultType.CONTACT_SAFETY_STOP: frozenset(),
    FaultType.TRAY_STATE_INCONSISTENT: frozenset(),
}

# Recovery strategy per fault: (strategy, rework stage, retry limit).
#
# Quality recovery is paired with a real reverse carrier route in ``runtime``:
# S2B→S2A for brazing defects and S4→the original install branch for fin
# defects.  The rollback stage below is therefore a physical process
# destination, not a state-only rewind.
#
# ``None`` means the unit is not rolled back at all (equipment-level faults).
_STRATEGIES: dict[FaultType, tuple[str, str | None, int]] = {
    FaultType.BRAZING_MISSING: ("LOCAL_BRAZING_REWORK", "MATERIAL_INSPECTION", 2),
    FaultType.BRAZING_PATH_DEVIATION: ("LOCAL_BRAZING_REWORK", "MATERIAL_INSPECTION", 2),
    FaultType.FIN_PICK_FAILED: ("MANUAL_REVIEW", None, 0),
    FaultType.FIN_GEOMETRY_FAILED: ("FIN_REINSTALL", "PRE_BRAZE_INSPECTION", 2),
    # Logistics/interlock faults hold the unit rather than rewinding a stage.
    FaultType.ELEVATOR_TIMEOUT: ("TRANSFER_SAFE_HOME_RETRY", None, 1),
    FaultType.FORK_TIMEOUT: ("TRANSFER_SAFE_HOME_RETRY", None, 1),
    FaultType.FURNACE_DOOR_INTERLOCK: ("FURNACE_INTERLOCK_RECHECK", None, 1),
    FaultType.FURNACE_PROFILE: ("MANUAL_REVIEW", None, 0),
    FaultType.ARM_UNAVAILABLE: ("MANUAL_REVIEW", None, 0),
    FaultType.RACK_LAYER_UNAVAILABLE: ("RACK_LAYER_REALLOCATION", None, 1),
    FaultType.CONTACT_SAFETY_STOP: ("MANUAL_REVIEW", None, 0),
    FaultType.TRAY_STATE_INCONSISTENT: ("MANUAL_REVIEW", None, 0),
}

SIMULATED_MANUAL_REVIEW_SECONDS = 10.0

# Extra work a rework costs, as a multiple of the reworked operation's nominal
# duration.  Restarting the interrupted operation alone is not enough: a braze
# gap has to be re-dispensed and re-inspected, which is strictly more work than
# the inspection that detected it.  Charging that cost is what makes recovery
# show up in makespan instead of looking free.
# Faults that stop a mechanism rather than spoiling a workpiece.  Isolating the
# resource is the whole effect: no unit is reworked, but the line waits.
#
# Only resources whose every start passes through ``_operation_start_allowed``
# may appear here.  ``FURNACE_DOOR`` and ``FURNACE_TRANSFER`` are deliberately
# excluded: several furnace branches commit door/tray state (``close_front``,
# ``open_rear``, ``mark_product_removed``) *before* calling ``_start``, so
# refusing the start would leave the furnace model and the stage machine
# disagreeing.  Those faults freeze the exact current operation on which they
# fire, then resume it after its safe retry interval.
_ISOLATED_MECHANISMS: dict[FaultType, tuple[str, ...]] = {
    FaultType.TRAY_STATE_INCONSISTENT: ("OUTPUT",),
}

# Faults modelled as a frozen-clock hold on the exact affected operation, used
# where isolating the mechanism would desynchronise committed physical state.
_HOLD_FAULTS: frozenset[FaultType] = frozenset(
    {
        FaultType.ELEVATOR_TIMEOUT,
        FaultType.FORK_TIMEOUT,
        FaultType.FURNACE_DOOR_INTERLOCK,
    }
)

_REWORK_EFFORT: dict[str, float] = {
    # A braze repair visits one affected path only; it is deliberately shorter
    # than repeating the complete multi-pass dispensing operation.
    "LOCAL_BRAZING_REWORK": 0.35,
    "FIN_REINSTALL": 1.5,
    "TRANSFER_SAFE_HOME_RETRY": 1.0,
    "FURNACE_INTERLOCK_RECHECK": 1.0,
}

_STRATEGY_LABELS_ZH: dict[str, str] = {
    "LOCAL_BRAZING_REWORK": "局部补涂并复检",
    "FIN_REINSTALL": "重新安装翅片并复检",
    "TRANSFER_SAFE_HOME_RETRY": "移载回零后重试一次",
    "FURNACE_INTERLOCK_RECHECK": "保持托盘锁定并复检炉门互锁",
    "RESOURCE_REALLOCATION": "隔离该资源并改派其他资源",
    "RACK_LAYER_REALLOCATION": "改派其他空闲炉层",
    "MANUAL_REVIEW": "转人工确认",
}


def _as_fault_type(value: FaultType | str) -> FaultType:
    """Coerce a name or enum member to :class:`FaultType`.

    ``FaultType`` is a plain ``Enum``, so ``str()`` on a member yields its repr
    rather than its value; coerce explicitly instead.
    """

    if isinstance(value, FaultType):
        return value
    return FaultType(str(value).strip().upper())


@dataclass(slots=True)
class PendingFaultRequest:
    """An armed manual injection awaiting a matching operation.

    Quality lifecycle ``ARMED → MANIFESTED → DETECTED → RECOVERED``.  Immediate
    equipment faults retain ``ARMED → FIRED``.
    """

    request_id: str
    fault_type: FaultType
    target: str
    severity: str
    auto_recover: bool
    duration_s: float | None
    label_zh: str
    armed_at: float
    visual_type: str = ""
    status: str = "ARMED"
    lifecycle_status: str = "ARMED"
    manifested_at: float | None = None
    detected_at: float | None = None
    recovered_at: float | None = None
    unit_id: str | None = None
    fired_at: float | None = None
    fault_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "fault_type": self.fault_type.value,
            "target": self.target,
            "severity": self.severity,
            "auto_recover": self.auto_recover,
            "duration_s": self.duration_s,
            "label_zh": self.label_zh,
            "visual_type": self.visual_type or self.fault_type.value,
            "armed_at": self.armed_at,
            "status": self.lifecycle_status,
            "legacy_status": self.status,
            "manifested_at": self.manifested_at,
            "detected_at": self.detected_at,
            "recovered_at": self.recovered_at,
            "unit_id": self.unit_id,
            "fired_at": self.fired_at,
            "fault_id": self.fault_id,
        }


@dataclass(slots=True)
class PhysicalFaultState:
    """A defect that exists in the workpiece before a camera discovers it."""

    defect_id: str
    request_id: str
    fault_type: FaultType
    visual_type: str
    target: str
    unit_id: str
    source_operation: str
    detection_operation: str
    manifested_at: float
    operation_index: int | None = None
    status: str = "MANIFESTED"
    detected_at: float | None = None
    repaired_at: float | None = None
    recovered_at: float | None = None
    fault_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "defect_id": self.defect_id,
            "request_id": self.request_id,
            "fault_type": self.fault_type.value,
            "visual_type": self.visual_type or self.fault_type.value,
            "target": self.target,
            "unit_id": self.unit_id,
            "source_operation": self.source_operation,
            "detection_operation": self.detection_operation,
            "operation_index": self.operation_index,
            "status": self.status,
            "manifested_at": self.manifested_at,
            "detected_at": self.detected_at,
            "repaired_at": self.repaired_at,
            "recovered_at": self.recovered_at,
            "fault_id": self.fault_id,
        }


@dataclass(slots=True)
class V2RecoveryPlan:
    """A recovery plan expressed as stage rollback rather than graph surgery."""

    recovery_id: str
    fault_id: str
    strategy: str
    label_zh: str
    unit_id: str | None
    rollback_stage: str | None
    recovery_class: str = "AUTONOMOUS_RECOVERY"
    final_disposition_zh: str = "修复后复检，合格则回归原订单"
    status: RecoveryStatus = RecoveryStatus.PLANNED
    retry_count: int = 0
    retry_limit: int = 1
    created_at: float = 0.0
    completed_at: float | None = None
    message: str = ""
    fault_label_zh: str = ""
    manual_review_started_at: float | None = None
    manual_review_complete_at: float | None = None
    # Set once the runtime has rewound the unit's stage machine, so a plan whose
    # rollback target equals the unit's current stage is not applied twice.
    rollback_applied: bool = False

    @property
    def effort_factor(self) -> float:
        """Rework duration as a multiple of the reworked operation's nominal."""

        return _REWORK_EFFORT.get(self.strategy, 1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recovery_id": self.recovery_id,
            "fault_id": self.fault_id,
            "strategy": self.strategy,
            "label_zh": self.label_zh,
            "unit_id": self.unit_id,
            "rollback_stage": self.rollback_stage,
            "recovery_class": self.recovery_class,
            "final_disposition_zh": self.final_disposition_zh,
            "rollback_applied": self.rollback_applied,
            "effort_factor": self.effort_factor,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "retry_limit": self.retry_limit,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "message": self.message,
            "fault_label_zh": self.fault_label_zh,
            "manual_review_started_at": self.manual_review_started_at,
            "manual_review_complete_at": self.manual_review_complete_at,
        }


@dataclass(slots=True)
class V2FaultController:
    """Owns V2's fault state: armed requests, records, plans and isolation.

    Deliberately holds no reference to MuJoCo or to the scene.  The runtime calls
    into it; the scene adapter reads its snapshot to drive visuals.
    """

    faults: dict[str, FaultRecord] = field(default_factory=dict)
    plans: dict[str, V2RecoveryPlan] = field(default_factory=dict)
    pending: dict[str, PendingFaultRequest] = field(default_factory=dict)
    physical_faults: dict[str, PhysicalFaultState] = field(default_factory=dict)
    # resource_id -> sim_time at which it auto-recovers (None = manual only)
    isolated_resources: dict[str, float | None] = field(default_factory=dict)
    unavailable_rack_layers: set[int] = field(default_factory=set)
    # unit_id -> extra seconds owed on its next operation (logistics holds)
    unit_holds: dict[str, float] = field(default_factory=dict)
    rework_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    cell_hold_until: float | None = None
    cell_hold_active: bool = False
    _fault_sequence: int = 0
    _request_sequence: int = 0
    _recovery_sequence: int = 0

    # ------------------------------------------------------------------ arming
    def arm(
        self,
        fault_type: FaultType | str,
        *,
        target: str,
        severity: str = "recoverable",
        auto_recover: bool = True,
        duration_s: float | None = None,
        label_zh: str = "",
        visual_type: str = "",
        now: float = 0.0,
    ) -> PendingFaultRequest:
        """Register a manual injection.

        Faults with no operation trigger fire immediately; the rest wait for a
        matching operation so the operator need not time the injection.
        """

        resolved = _as_fault_type(fault_type)
        self._request_sequence += 1
        request = PendingFaultRequest(
            request_id=f"V2FREQ_{self._request_sequence:04d}",
            fault_type=resolved,
            target=str(target),
            severity=str(severity),
            auto_recover=bool(auto_recover),
            duration_s=None if duration_s is None else float(duration_s),
            label_zh=label_zh or resolved.value,
            armed_at=float(now),
            visual_type=str(visual_type or resolved.value).strip().upper(),
        )
        self.pending[request.request_id] = request
        if not FAULT_OPERATION_KINDS.get(resolved):
            # Equipment/safety faults act on the cell, not on a process step.
            record = self.inject(
                resolved,
                source=request.target or "SYSTEM",
                target=request.target,
                now=now,
                severity=request.severity,
                auto_recover=request.auto_recover,
                duration_s=request.duration_s,
                label_zh=request.label_zh,
                visual_type=request.visual_type,
            )
            request.status = "FIRED"
            request.lifecycle_status = "FIRED"
            request.fired_at = float(now)
            request.fault_id = record.fault_id
        return request

    def manifest_matching(
        self,
        operations: Iterable[Any],
        *,
        now: float,
        unit_lookup: Mapping[str, Any] | None = None,
    ) -> list[PhysicalFaultState]:
        """Create latent physical defects during their producing operation."""

        manifested: list[PhysicalFaultState] = []
        for request in self.pending.values():
            if request.status != "ARMED":
                continue
            kinds = FAULT_MANIFEST_KINDS.get(request.fault_type, frozenset())
            for operation in operations:
                if operation.kind not in kinds:
                    continue
                if not self._target_matches(request, operation, unit_lookup, exact_index=True):
                    continue
                operation_index = None
                if request.target.startswith("fin_"):
                    operation_index = self._target_index(request.target)
                elif request.target.startswith(("slot_", "path_")):
                    operation_index = self._target_path_index(request.target)
                detection = sorted(FAULT_DETECTION_KINDS.get(request.fault_type, frozenset()))
                defect = PhysicalFaultState(
                    defect_id=f"V2DEFECT_{request.request_id.removeprefix('V2FREQ_')}",
                    request_id=request.request_id,
                    fault_type=request.fault_type,
                    visual_type=request.visual_type,
                    target=request.target,
                    unit_id=operation.unit_id,
                    source_operation=operation.kind,
                    detection_operation=detection[0] if detection else "",
                    manifested_at=float(now),
                    operation_index=operation_index,
                )
                self.physical_faults[defect.defect_id] = defect
                request.status = "MANIFESTED"
                request.lifecycle_status = "MANIFESTED"
                request.manifested_at = float(now)
                request.unit_id = operation.unit_id
                manifested.append(defect)
                break
        return manifested

    def detect_for_operation(
        self,
        operation: Any,
        *,
        now: float,
        unit_lookup: Mapping[str, Any] | None = None,
    ) -> list[FaultRecord]:
        """Let a completed camera operation discover existing defects."""

        detected: list[FaultRecord] = []
        for defect in self.physical_faults.values():
            if defect.status != "MANIFESTED" or defect.unit_id != operation.unit_id:
                continue
            if operation.kind not in FAULT_DETECTION_KINDS.get(defect.fault_type, frozenset()):
                continue
            request = self.pending[defect.request_id]
            record = self.inject(
                defect.fault_type,
                source=operation.resource,
                target=defect.target,
                unit_id=operation.unit_id,
                now=now,
                severity=request.severity,
                auto_recover=request.auto_recover,
                duration_s=request.duration_s,
                label_zh=request.label_zh,
                visual_type=request.visual_type,
            )
            defect.status = "DETECTED"
            defect.detected_at = float(now)
            defect.fault_id = record.fault_id
            # ``FIRED`` keeps compatibility with the original public object
            # contract; the snapshot/UI receives the precise lifecycle state.
            request.status = "FIRED"
            request.lifecycle_status = "DETECTED"
            request.detected_at = float(now)
            request.fired_at = float(now)
            request.fault_id = record.fault_id
            detected.append(record)
        return detected

    def fire_matching(
        self,
        operations: Iterable[Any],
        *,
        now: float,
        unit_lookup: Mapping[str, Any] | None = None,
    ) -> list[FaultRecord]:
        """Compatibility wrapper for non-quality deferred faults.

        Quality faults are intentionally handled by ``manifest_matching`` and
        ``detect_for_operation`` so they cannot appear for the first time during
        repair.  This method remains for external callers and legacy tests.
        """

        fired: list[FaultRecord] = []
        for request in list(self.pending.values()):
            if request.status != "ARMED":
                continue
            if request.fault_type in FAULT_MANIFEST_KINDS:
                continue
            kinds = FAULT_OPERATION_KINDS.get(request.fault_type, frozenset())
            if not kinds:
                continue
            for operation in operations:
                if operation.kind not in kinds:
                    continue
                if not self._target_matches(request, operation, unit_lookup):
                    continue
                record = self.inject(
                    request.fault_type,
                    source=operation.resource,
                    target=request.target,
                    unit_id=operation.unit_id,
                    now=now,
                    severity=request.severity,
                    auto_recover=request.auto_recover,
                    duration_s=request.duration_s,
                    label_zh=request.label_zh,
                    visual_type=request.visual_type,
                )
                request.status = "FIRED"
                request.lifecycle_status = "DETECTED"
                request.fired_at = float(now)
                request.fault_id = record.fault_id
                fired.append(record)
                break
        return fired

    @staticmethod
    def _target_matches(
        request: PendingFaultRequest,
        operation: Any,
        unit_lookup: Mapping[str, Any] | None,
        *,
        exact_index: bool = False,
    ) -> bool:
        target = request.target.strip()
        if not target:
            return True
        if target.upper() in {"ARM1", "ARM2", "ARM3"}:
            return operation.resource == target.upper()
        if target.startswith("fin_"):
            if not exact_index:
                return True
            unit = None if unit_lookup is None else unit_lookup.get(operation.unit_id)
            return (
                unit is not None
                and operation.kind == "INSTALL_FIN"
                and int(getattr(unit, "fins_installed", 0)) + 1 == V2FaultController._target_index(target)
            )
        if target.startswith(("slot_", "path_")):
            if exact_index and operation.kind == "DISPENSING":
                unit = None if unit_lookup is None else unit_lookup.get(operation.unit_id)
                path_index = V2FaultController._target_path_index(target)
                pass_count = max(1, int(getattr(unit, "fin_count", 1)))
                if path_index is None:
                    return False
                target_pass = max(0, (path_index - 1) // 2)
                duration = max(float(getattr(operation, "duration_s", 0.0)), 1.0e-9)
                elapsed_fraction = 1.0 - float(getattr(operation, "remaining_s", 0.0)) / duration
                return elapsed_fraction >= target_pass / pass_count
            return True
        unit = None if unit_lookup is None else unit_lookup.get(operation.unit_id)
        if unit is not None and target.isdigit():
            return getattr(unit, "furnace_layer", None) == int(target)
        return True

    @staticmethod
    def _target_index(target: str) -> int | None:
        try:
            return int(str(target).split("_", 2)[1])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _target_path_index(target: str) -> int | None:
        """Map both V2 ``path_XX`` and V1 ``slot_XX_side`` to one bead index."""

        value = str(target).lower()
        index = V2FaultController._target_index(value)
        if index is None or value.startswith("path_"):
            return index
        if not value.startswith("slot_"):
            return None
        side = value.split("_", 2)[2] if len(value.split("_", 2)) > 2 else "left"
        return 2 * index - (1 if side == "left" else 0)

    def expire_armed(self) -> None:
        """Mark still-armed requests as missed once a run has terminated."""

        for request in self.pending.values():
            if request.status == "ARMED":
                request.status = "MISSED"

    # --------------------------------------------------------------- injection
    def inject(
        self,
        fault_type: FaultType | str,
        *,
        source: str,
        target: str = "",
        unit_id: str | None = None,
        now: float = 0.0,
        severity: str = "recoverable",
        auto_recover: bool = True,
        duration_s: float | None = None,
        label_zh: str = "",
        visual_type: str = "",
    ) -> FaultRecord:
        """Record a fault and plan its recovery."""

        resolved = _as_fault_type(fault_type)
        self._fault_sequence += 1
        recoverable = severity != "severe" and resolved not in {
            FaultType.CONTACT_SAFETY_STOP,
            FaultType.TRAY_STATE_INCONSISTENT,
        }
        record = FaultRecord(
            fault_id=f"V2FAULT_{self._fault_sequence:04d}",
            fault_type=resolved,
            source=str(source or "SYSTEM"),
            related_task_id=unit_id,
            sim_time=float(now),
            recoverable=recoverable,
            details={
                "target": target,
                "severity": severity,
                "unit_id": unit_id,
                "auto_recover": bool(auto_recover),
                "duration_s": duration_s,
                "label_zh": str(label_zh or resolved.value),
                "visual_type": str(visual_type or resolved.value).strip().upper(),
            },
        )
        self.faults[record.fault_id] = record

        # Manual-review faults do not use the generic resource timeout.  Their
        # release is owned by the single ten-second review state machine below,
        # so the popup, resource isolation and order continuation stay atomic.
        simulated_manual_review = self._uses_simulated_manual_review(record)
        deadline = (
            float(now) + float(duration_s)
            if auto_recover and duration_s and record.recoverable and not simulated_manual_review
            else None
        )
        if resolved is FaultType.ARM_UNAVAILABLE:
            self.isolated_resources[(target or source).upper()] = deadline
        elif resolved is FaultType.RACK_LAYER_UNAVAILABLE and target.isdigit():
            self.unavailable_rack_layers.add(int(target))
        elif resolved in _ISOLATED_MECHANISMS:
            # Stop the mechanism itself; no unit is reworked but the line waits.
            for resource in _ISOLATED_MECHANISMS[resolved]:
                self.isolated_resources[resource] = deadline
        elif resolved in _HOLD_FAULTS and unit_id:
            # Charge the delay to the affected unit's next operation, because
            # isolating these mechanisms would desynchronise committed state.
            self.unit_holds[unit_id] = self.unit_holds.get(unit_id, 0.0) + float(duration_s or 4.0)
        if resolved is FaultType.CONTACT_SAFETY_STOP:
            self.cell_hold_active = True
            self.cell_hold_until = deadline

        self._plan_recovery(record, now=now)
        return record

    def _plan_recovery(self, record: FaultRecord, *, now: float) -> V2RecoveryPlan:
        strategy, rollback, retry_limit = _STRATEGIES.get(record.fault_type, ("MANUAL_REVIEW", None, 0))
        unit_id = record.details.get("unit_id")
        target = str(record.details.get("target", ""))
        visual_type = str(record.details.get("visual_type", record.fault_type.value)).upper()
        definition = MANUAL_FAULT_CATALOG.get(visual_type) or MANUAL_FAULT_CATALOG.get(
            record.fault_type.value
        )

        # FIN_POSE is the first catalogue item and keeps its automatic Arm3
        # correction.  The sixth item uses the same typed runtime fault but is
        # deliberately presented as FIN_GEOMETRY_FAILED and now goes to the
        # simulated human review requested by the operator.
        if record.fault_type is FaultType.FIN_GEOMETRY_FAILED and visual_type == "FIN_POSE":
            strategy, rollback, retry_limit = "FIN_REINSTALL", "PRE_BRAZE_INSPECTION", 2
        elif self._uses_simulated_manual_review(record):
            strategy, rollback, retry_limit = "MANUAL_REVIEW", None, 0

        # Per-(strategy, unit, target) retry ceiling, as in V1: after the limit
        # the fault stops being auto-recoverable and goes to a human.
        retry_bounded = rollback is not None or strategy in {
            "TRANSFER_SAFE_HOME_RETRY",
            "FURNACE_INTERLOCK_RECHECK",
        }
        if retry_bounded and unit_id:
            key = (f"{record.fault_type.value}:{strategy}:{unit_id}", target)
            attempts = self.rework_counts.get(key, 0)
            if attempts >= retry_limit:
                record.recoverable = False
                strategy, rollback = "MANUAL_REVIEW", None
            else:
                self.rework_counts[key] = attempts + 1

        if not record.recoverable:
            strategy, rollback = "MANUAL_REVIEW", None

        simulated_manual_review = strategy == "MANUAL_REVIEW"
        recovery_class = "MANUAL_DISPOSITION" if simulated_manual_review else "AUTONOMOUS_RECOVERY"
        final_disposition_zh = (
            definition.final_disposition_zh
            if definition is not None
            else "人工确认后恢复安全状态" if simulated_manual_review else "修复后复检，合格则回归原订单"
        )
        fault_label_zh = str(record.details.get("label_zh") or record.fault_type.value)
        self._recovery_sequence += 1
        plan = V2RecoveryPlan(
            recovery_id=f"V2REC_{self._recovery_sequence:04d}",
            fault_id=record.fault_id,
            strategy=strategy,
            label_zh=_STRATEGY_LABELS_ZH.get(strategy, strategy),
            unit_id=unit_id,
            rollback_stage=rollback,
            recovery_class=recovery_class,
            final_disposition_zh=final_disposition_zh,
            status=(RecoveryStatus.MANUAL_REVIEW if strategy == "MANUAL_REVIEW" else RecoveryStatus.RUNNING),
            retry_limit=max(1, retry_limit),
            created_at=float(now),
            message=(
                ""
                if strategy != "MANUAL_REVIEW"
                else f"发生{fault_label_zh}故障❌，需进行人工审核🔩🔧，请稍作等待⏰"
            ),
            fault_label_zh=fault_label_zh,
            manual_review_started_at=(float(now) if strategy == "MANUAL_REVIEW" else None),
            manual_review_complete_at=(
                float(now) + SIMULATED_MANUAL_REVIEW_SECONDS if simulated_manual_review else None
            ),
        )
        self.plans[plan.recovery_id] = plan
        record.recovery_id = plan.recovery_id
        return plan

    @staticmethod
    def _uses_simulated_manual_review(record: FaultRecord) -> bool:
        visual_type = str(record.details.get("visual_type", record.fault_type.value)).upper()
        return record.fault_type in {
            FaultType.FURNACE_PROFILE,
            FaultType.FIN_PICK_FAILED,
            FaultType.ARM_UNAVAILABLE,
            FaultType.CONTACT_SAFETY_STOP,
            FaultType.TRAY_STATE_INCONSISTENT,
        } or (record.fault_type is FaultType.FIN_GEOMETRY_FAILED and visual_type != "FIN_POSE")

    @staticmethod
    def _manual_review_message(record: FaultRecord) -> str:
        label = str(record.details.get("label_zh") or record.fault_type.value)
        return f"发生{label}故障❌，需进行人工审核🔩🔧，请稍作等待⏰"

    def _begin_manual_review(
        self,
        plan: V2RecoveryPlan,
        record: FaultRecord,
        now: float,
    ) -> None:
        """Enter the common, idempotent ten-second simulated review window."""

        if plan.status is RecoveryStatus.MANUAL_REVIEW and plan.manual_review_complete_at is not None:
            return
        plan.status = RecoveryStatus.MANUAL_REVIEW
        plan.manual_review_started_at = float(now)
        plan.manual_review_complete_at = float(now) + SIMULATED_MANUAL_REVIEW_SECONDS
        plan.fault_label_zh = str(record.details.get("label_zh") or record.fault_type.value)
        plan.message = self._manual_review_message(record)

    def _release_manual_review_isolation(self, record: FaultRecord) -> None:
        """Release exactly the physical/logical hold owned by one reviewed fault."""

        if record.fault_type is FaultType.ARM_UNAVAILABLE:
            resource = str(record.details.get("target") or record.source).upper()
            self.isolated_resources.pop(resource, None)
        if record.fault_type is FaultType.CONTACT_SAFETY_STOP:
            self.cell_hold_active = False
            self.cell_hold_until = None
        for resource in _ISOLATED_MECHANISMS.get(record.fault_type, ()):
            self.isolated_resources.pop(resource, None)
        if record.fault_type is FaultType.RACK_LAYER_UNAVAILABLE:
            target = str(record.details.get("target", ""))
            if target.isdigit():
                self.unavailable_rack_layers.discard(int(target))

    def service_manual_reviews(self, now: float) -> list[V2RecoveryPlan]:
        """Complete the explicitly simulated ten-second human repair window."""

        completed: list[V2RecoveryPlan] = []
        for plan in self.plans.values():
            deadline = plan.manual_review_complete_at
            if (
                plan.status is not RecoveryStatus.MANUAL_REVIEW
                or deadline is None
                or float(now) + 1.0e-9 < deadline
            ):
                continue
            record = self.faults.get(plan.fault_id)
            if record is None:
                continue
            self._release_manual_review_isolation(record)
            record.recovered = True
            plan.status = RecoveryStatus.SUCCEEDED
            plan.completed_at = float(now)
            plan.message = "修改成功✅"
            for request in self.pending.values():
                if request.fault_id != record.fault_id:
                    continue
                request.status = "RECOVERED"
                request.lifecycle_status = "RECOVERED"
                request.recovered_at = float(now)
            for defect in self.physical_faults.values():
                if defect.fault_id != record.fault_id:
                    continue
                defect.status = "RECOVERED"
                defect.repaired_at = float(now)
                defect.recovered_at = float(now)
                request = self.pending.get(defect.request_id)
                if request is not None:
                    request.status = "RECOVERED"
                    request.lifecycle_status = "RECOVERED"
                    request.recovered_at = float(now)
            completed.append(plan)
        return completed

    # ---------------------------------------------------------------- recovery
    def recover_resource(self, resource_id: str, now: float) -> bool:
        """Bring an isolated resource back online."""

        resource = str(resource_id).upper()
        if resource not in self.isolated_resources:
            return False
        for record in self.faults.values():
            if (
                record.fault_type is not FaultType.ARM_UNAVAILABLE
                or record.recovered
                or str(record.details.get("target", record.source)).upper() != resource
            ):
                continue
            plan = self.plans.get(record.recovery_id or "")
            if (
                plan is not None
                and plan.status is RecoveryStatus.MANUAL_REVIEW
                and plan.manual_review_complete_at is not None
                and float(now) + 1.0e-9 < plan.manual_review_complete_at
            ):
                return False
        del self.isolated_resources[resource]
        for record in self.faults.values():
            if (
                record.fault_type is FaultType.ARM_UNAVAILABLE
                and not record.recovered
                and str(record.details.get("target", record.source)).upper() == resource
            ):
                record.recovered = True
                plan = self.plans.get(record.recovery_id or "")
                if plan is not None and plan.status is not RecoveryStatus.MANUAL_REVIEW:
                    plan.status = RecoveryStatus.SUCCEEDED
                    plan.completed_at = float(now)
        return True

    def release_rack_layer(self, layer_index: int, now: float) -> bool:
        if layer_index not in self.unavailable_rack_layers:
            return False
        self.unavailable_rack_layers.discard(layer_index)
        for record in self.faults.values():
            if (
                record.fault_type is FaultType.RACK_LAYER_UNAVAILABLE
                and not record.recovered
                and str(record.details.get("target")) == str(layer_index)
            ):
                record.recovered = True
                plan = self.plans.get(record.recovery_id or "")
                if plan is not None and plan.status is not RecoveryStatus.MANUAL_REVIEW:
                    plan.status = RecoveryStatus.SUCCEEDED
                    plan.completed_at = float(now)
        return True

    def mark_rack_reallocated(
        self,
        fault_layer: int,
        selected_layer: int,
        unit_id: str,
        now: float,
    ) -> bool:
        """Complete the production recovery without pretending the rack is repaired.

        The failed layer remains unavailable (and visibly faulted) until
        :meth:`release_rack_layer` is called.  Selecting another empty layer is
        nevertheless a completed recovery plan for the affected product.
        """

        if int(selected_layer) == int(fault_layer):
            return False
        changed = False
        for record in self.faults.values():
            if record.fault_type is not FaultType.RACK_LAYER_UNAVAILABLE or str(
                record.details.get("target")
            ) != str(fault_layer):
                continue
            plan = self.plans.get(record.recovery_id or "")
            if plan is None or plan.status is not RecoveryStatus.RUNNING:
                continue
            record.details.update(
                {
                    "mitigated": True,
                    "reallocated_unit_id": str(unit_id),
                    "selected_layer": int(selected_layer),
                }
            )
            plan.status = RecoveryStatus.SUCCEEDED
            plan.completed_at = float(now)
            plan.message = f"故障层{int(fault_layer)}保持隔离，" f"托盘已改派至第{int(selected_layer)}层"
            changed = True
        return changed

    def bind_hold_to_operation(
        self,
        fault_id: str,
        *,
        resource: str,
        operation_kind: str,
        seconds: float,
    ) -> bool:
        """Bind a mechanism timeout to the operation on which it fired."""

        record = self.faults.get(str(fault_id))
        if record is None or record.fault_type not in _HOLD_FAULTS:
            return False
        record.details.update(
            {
                "affected_resource": str(resource),
                "affected_operation": str(operation_kind),
                "hold_seconds": float(seconds),
            }
        )
        plan = self.plans.get(record.recovery_id or "")
        if plan is not None:
            plan.message = f"当前工序安全停止 {float(seconds):.1f}s，随后原工序重试"
        return True

    def complete_bound_operation_recovery(
        self,
        fault_ids: Iterable[str],
        *,
        resource: str,
        unit_id: str,
        operation_kind: str,
        now: float,
    ) -> tuple[str, ...]:
        """Close only timeout plans bound to the operation that just recovered."""

        completed: list[str] = []
        for fault_id in fault_ids:
            record = self.faults.get(str(fault_id))
            if (
                record is None
                or record.recovered
                or record.fault_type not in _HOLD_FAULTS
                or str(record.details.get("unit_id")) != str(unit_id)
                or str(record.details.get("affected_resource")) != str(resource)
                or str(record.details.get("affected_operation")) != str(operation_kind)
            ):
                continue
            plan = self.plans.get(record.recovery_id or "")
            if plan is None or plan.status is not RecoveryStatus.RUNNING:
                continue
            record.recovered = True
            plan.status = RecoveryStatus.SUCCEEDED
            plan.completed_at = float(now)
            plan.message = "机构已完成安全复位并重试原工序"
            completed.append(record.fault_id)
        return tuple(completed)

    def service_auto_recovery(self, now: float) -> list[str]:
        """Release resources whose auto-recovery deadline has passed."""

        due = [
            resource
            for resource, deadline in self.isolated_resources.items()
            if deadline is not None and now >= deadline
        ]
        for resource in due:
            self.recover_resource(resource, now)
        if self.cell_hold_active and self.cell_hold_until is not None and now >= self.cell_hold_until:
            self.cell_hold_active = False
            self.cell_hold_until = None
            due.append("CELL_SAFETY_HOLD")
            for record in self.faults.values():
                if record.fault_type is not FaultType.CONTACT_SAFETY_STOP or record.recovered:
                    continue
                record.recovered = True
                plan = self.plans.get(record.recovery_id or "")
                if plan is not None and plan.status is not RecoveryStatus.MANUAL_REVIEW:
                    plan.status = RecoveryStatus.SUCCEEDED
                    plan.completed_at = float(now)
        return due

    def complete_recovery(self, unit_id: str, now: float) -> None:
        """Mark a unit's recovery succeeded once the rework has actually run.

        Only plans whose rollback was applied count: a plan that never rewound
        anything has not recovered from anything.
        """

        for plan in self.plans.values():
            if plan.unit_id != unit_id or plan.status is not RecoveryStatus.RUNNING:
                continue
            if plan.rollback_stage and not plan.rollback_applied:
                continue
            plan.status = RecoveryStatus.SUCCEEDED
            plan.completed_at = float(now)
            record = self.faults.get(plan.fault_id)
            if record is not None:
                record.recovered = True
            for defect in self.physical_faults.values():
                if defect.fault_id != plan.fault_id:
                    continue
                defect.status = "RECOVERED"
                defect.recovered_at = float(now)
                request = self.pending.get(defect.request_id)
                if request is not None:
                    request.status = "RECOVERED"
                    request.lifecycle_status = "RECOVERED"
                    request.recovered_at = float(now)

    def mark_repaired(self, unit_id: str, now: float) -> None:
        """Clear the physical defect after visible rework, before reinspection."""

        for defect in self.physical_faults.values():
            if defect.unit_id != unit_id or defect.status != "DETECTED":
                continue
            defect.status = "REPAIRED"
            defect.repaired_at = float(now)
            request = self.pending.get(defect.request_id)
            if request is not None:
                request.status = "RECOVERING"
                request.lifecycle_status = "RECOVERING"

    def action(self, recovery_id: str, action: str, now: float) -> bool:
        """UI verbs: pause / resume / retry / manual_review."""

        plan = self.plans.get(str(recovery_id))
        if plan is None:
            return False
        verb = str(action).lower()
        if verb == "pause" and plan.status is RecoveryStatus.RUNNING:
            plan.status = RecoveryStatus.PAUSED
        elif verb == "resume" and plan.status is RecoveryStatus.PAUSED:
            plan.status = RecoveryStatus.RUNNING
        elif verb == "retry":
            record = self.faults.get(plan.fault_id)
            if (
                plan.status is RecoveryStatus.MANUAL_REVIEW
                and plan.manual_review_complete_at is not None
                and float(now) + 1.0e-9 < plan.manual_review_complete_at
            ):
                return False
            # For safety faults, Retry means that the operator has inspected
            # the cell and explicitly confirmed it is safe to release.  It is
            # not an automatic process retry and therefore completes the manual
            # recovery instead of returning to an unowned RUNNING state.
            if (
                plan.status is RecoveryStatus.MANUAL_REVIEW
                and record is not None
                and record.fault_type in {FaultType.CONTACT_SAFETY_STOP, FaultType.TRAY_STATE_INCONSISTENT}
            ):
                if record.fault_type is FaultType.CONTACT_SAFETY_STOP:
                    self.cell_hold_active = False
                    self.cell_hold_until = None
                else:
                    for resource in _ISOLATED_MECHANISMS[record.fault_type]:
                        self.isolated_resources.pop(resource, None)
                record.recovered = True
                plan.retry_count += 1
                plan.status = RecoveryStatus.SUCCEEDED
                plan.completed_at = float(now)
                plan.message = "操作员已完成安全核验，故障隔离已释放"
                return True
            if (
                plan.status is RecoveryStatus.MANUAL_REVIEW
                and record is not None
                and record.fault_type in _HOLD_FAULTS
            ):
                record.recovered = True
                plan.retry_count += 1
                plan.status = RecoveryStatus.SUCCEEDED
                plan.completed_at = float(now)
                plan.message = "操作员已检修机构并确认可继续当前工序"
                return True
            if plan.status is RecoveryStatus.MANUAL_REVIEW:
                plan.message = "该质量故障需要人工处置结论，不能伪装为自动重试"
                return False
            if plan.retry_count >= plan.retry_limit:
                if record is not None:
                    self._begin_manual_review(plan, record, now)
                return False
            plan.retry_count += 1
            plan.status = RecoveryStatus.RUNNING
        elif verb == "manual_review":
            record = self.faults.get(plan.fault_id)
            if record is None:
                return False
            self._begin_manual_review(plan, record, now)
        else:
            return False
        return True

    # ---------------------------------------------------------------- queries
    def take_hold(self, unit_id: str, seconds: float | None = None) -> float:
        """Consume one fault's delay, or all delay owed by a unit."""

        owed = self.unit_holds.get(unit_id, 0.0)
        if seconds is None:
            self.unit_holds.pop(unit_id, None)
            return owed
        consumed = min(owed, max(0.0, float(seconds)))
        remaining = max(0.0, owed - consumed)
        if remaining <= 1.0e-12:
            self.unit_holds.pop(unit_id, None)
        else:
            self.unit_holds[unit_id] = remaining
        return consumed

    def resource_available(self, resource_id: str) -> bool:
        return str(resource_id).upper() not in self.isolated_resources

    def layer_available(self, layer_index: int) -> bool:
        return layer_index not in self.unavailable_rack_layers

    def cell_available(self) -> bool:
        return not self.cell_hold_active

    def pending_rollback(self) -> dict[str, str]:
        """unit_id -> stage to roll back to, for plans awaiting application."""

        return {
            plan.unit_id: plan.rollback_stage
            for plan in self.plans.values()
            if plan.unit_id and plan.rollback_stage and plan.status is RecoveryStatus.RUNNING
        }

    def reset(self) -> None:
        self.faults.clear()
        self.plans.clear()
        self.pending.clear()
        self.physical_faults.clear()
        self.isolated_resources.clear()
        self.unavailable_rack_layers.clear()
        self.unit_holds.clear()
        self.rework_counts.clear()
        self.cell_hold_until = None
        self.cell_hold_active = False
        self._fault_sequence = 0
        self._request_sequence = 0
        self._recovery_sequence = 0

    def snapshot(self) -> dict[str, Any]:
        notices = [
            {
                "recovery_id": plan.recovery_id,
                "fault_label_zh": plan.fault_label_zh,
                "status": plan.status.value,
                "message": plan.message,
                "started_at": plan.manual_review_started_at,
                "complete_at": plan.manual_review_complete_at,
            }
            for plan in self.plans.values()
            if plan.manual_review_started_at is not None
        ]
        catalog = tuple(MANUAL_FAULT_CATALOG.values())
        unresolved = [record for record in self.faults.values() if not record.recovered]
        active_plans = [
            plan
            for plan in self.plans.values()
            if plan.status not in {RecoveryStatus.SUCCEEDED, RecoveryStatus.FAILED}
        ]
        return {
            "faults_v2": [record.as_dict() for record in self.faults.values()],
            "recoveries": [plan.as_dict() for plan in self.plans.values()],
            "manual_fault_requests": [item.as_dict() for item in self.pending.values()],
            "physical_faults": [item.as_dict() for item in self.physical_faults.values()],
            "isolated_resources": dict(sorted(self.isolated_resources.items())),
            "unavailable_rack_layers": sorted(self.unavailable_rack_layers),
            "cell_safety_hold": {
                "active": self.cell_hold_active,
                "recover_at": self.cell_hold_until,
            },
            "manual_review_notices": notices,
            "fault_policy_summary": {
                "catalog_count": len(catalog),
                "autonomous_count": sum(item.recovery_class == "AUTONOMOUS_RECOVERY" for item in catalog),
                "manual_count": sum(item.recovery_class == "MANUAL_DISPOSITION" for item in catalog),
                "active_autonomous": sum(
                    item.recovery_class == "AUTONOMOUS_RECOVERY" for item in active_plans
                ),
                "active_manual": sum(item.recovery_class == "MANUAL_DISPOSITION" for item in active_plans),
                "unresolved_count": len(unresolved),
            },
            "fault_count": len(self.faults),
            "recovered_fault_count": sum(1 for record in self.faults.values() if record.recovered),
        }


__all__ = [
    "FAULT_DETECTION_KINDS",
    "FAULT_MANIFEST_KINDS",
    "FAULT_OPERATION_KINDS",
    "PendingFaultRequest",
    "PhysicalFaultState",
    "V2FaultController",
    "V2RecoveryPlan",
]
