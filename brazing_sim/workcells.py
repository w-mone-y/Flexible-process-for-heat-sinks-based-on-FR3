"""Domain state for the shallow-U asynchronous pallet line.

The classes in this module deliberately have no MuJoCo dependency.  They are
the authoritative ownership/interlock layer used by the scheduler, the
physical actors and the UI.  Geometry may follow this state, but must never
invent a second copy of a tray or changeover module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Iterable

from .profiles import quintic_time_scaling


class CellValue(str, Enum):
    def __str__(self) -> str:
        return self.value


class WorkstationId(CellValue):
    S1_BASE_LOADING = "S1_BASE_LOADING"
    S2A_DISPENSING = "S2A_DISPENSING"
    S2B_MATERIAL_INSPECTION = "S2B_MATERIAL_INSPECTION"
    S3_FIN_ASSEMBLY = "S3_FIN_ASSEMBLY"
    RACK_INFEED = "RACK_INFEED"
    POST_BRAZE_INSPECTION = "POST_BRAZE_INSPECTION"
    # Historical identifiers remain deserializable but are not emitted by
    # the current runtime.
    TABLE2_ASSEMBLY = "TABLE2_ASSEMBLY"
    TABLE2_PROCESS = "TABLE2_PROCESS"


class TransferId(CellValue):
    S1_S2A = "TRANSFER_S1_S2A"
    S2A_S2B = "TRANSFER_S2A_S2B"
    S2B_S3 = "TRANSFER_S2B_S3"
    S3_RACK = "TRANSFER_S3_RACK"


class TransferStatus(CellValue):
    IDLE = "IDLE"
    RESERVED = "RESERVED"
    MOVING = "MOVING"
    SETTLING = "SETTLING"
    RETURNING = "RETURNING"
    FAULTED = "FAULTED"


class TrayOwner(CellValue):
    EMPTY_BUFFER = "EMPTY_BUFFER"
    STATION_S1 = "STATION_S1"
    TRANSFER_S1_S2A = "TRANSFER_S1_S2A"
    STATION_S2A = "STATION_S2A"
    TRANSFER_S2A_S2B = "TRANSFER_S2A_S2B"
    STATION_S2B = "STATION_S2B"
    TRANSFER_S2B_S3 = "TRANSFER_S2B_S3"
    STATION_S3 = "STATION_S3"
    TRANSFER_S3_RACK = "TRANSFER_S3_RACK"
    RACK_INFEED = "RACK_INFEED"
    ELEVATOR = "ELEVATOR"
    FURNACE_RACK = "FURNACE_RACK"
    POST_INSPECTION = "POST_INSPECTION"
    OUTPUT = "OUTPUT"


@dataclass(slots=True)
class AsyncTransferState:
    transfer_id: TransferId
    source: WorkstationId
    target: WorkstationId
    tray_id: str | None = None
    status: TransferStatus = TransferStatus.IDLE
    progress: float = 0.0
    started_at: float | None = None
    settled_at: float | None = None

    def reserve(self, tray_id: str, now: float) -> None:
        if self.status is not TransferStatus.IDLE or self.tray_id is not None:
            raise RuntimeError(f"{self.transfer_id.value} is already occupied")
        self.tray_id = str(tray_id)
        self.status = TransferStatus.RESERVED
        self.progress = 0.0
        self.started_at = float(now)
        self.settled_at = None

    def set_progress(self, progress: float) -> None:
        self.progress = min(1.0, max(0.0, float(progress)))
        self.status = TransferStatus.MOVING if self.progress < 1.0 else TransferStatus.SETTLING

    def release(self) -> str:
        if self.tray_id is None:
            raise RuntimeError(f"{self.transfer_id.value} has no pallet to release")
        tray_id = self.tray_id
        self.tray_id = None
        self.status = TransferStatus.IDLE
        self.progress = 0.0
        self.started_at = None
        self.settled_at = None
        return tray_id

    def as_dict(self) -> dict[str, object]:
        return {
            "transfer_id": self.transfer_id.value,
            "source": self.source.value,
            "target": self.target.value,
            "tray_id": self.tray_id,
            "status": self.status.value,
            "progress": self.progress,
            "started_at": self.started_at,
            "settled_at": self.settled_at,
        }


class NestId(CellValue):
    NEST_A = "NEST_A"
    NEST_B = "NEST_B"


class TrayRoutePhase(CellValue):
    EMPTY_BUFFER = "EMPTY_BUFFER"
    CHANGEOVER = "CHANGEOVER"
    MOLD_READY = "MOLD_READY"
    BASE_READY = "BASE_READY"
    MATERIAL_READY = "MATERIAL_READY"
    ASSEMBLY_READY = "ASSEMBLY_READY"
    LOCKED = "LOCKED"
    OUTFEED = "OUTFEED"
    FURNACE = "FURNACE"
    FINISHED_GOODS = "FINISHED_GOODS"
    RETURNING = "RETURNING"


class ModuleKind(CellValue):
    MOLD = "MOLD"
    FRONT_COMB = "FRONT_COMB"
    REAR_COMB = "REAR_COMB"
    FRONT_PRESS = "FRONT_PRESS"
    REAR_PRESS = "REAR_PRESS"


class ModuleLocation(CellValue):
    RACK = "RACK"
    GANTRY = "GANTRY"
    STAGING = "STAGING"
    TRAY = "TRAY"


class ChangeoverStatus(CellValue):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(slots=True)
class WorkstationState:
    station_id: WorkstationId
    world_xy: tuple[float, float]
    capabilities: tuple[str, ...]
    nest_id: NestId | None = None
    tray_id: str | None = None
    occupied_by: str | None = None
    safe_for_transfer: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "station_id": self.station_id.value,
            "world_xy": list(self.world_xy),
            "capabilities": list(self.capabilities),
            "nest_id": None if self.nest_id is None else self.nest_id.value,
            "tray_id": self.tray_id,
            "occupied_by": self.occupied_by,
            "safe_for_transfer": self.safe_for_transfer,
        }


@dataclass(slots=True)
class TrayRouteState:
    tray_id: str
    phase: TrayRoutePhase = TrayRoutePhase.EMPTY_BUFFER
    owner: TrayOwner = TrayOwner.EMPTY_BUFFER
    nest_id: NestId | None = None
    station_id: WorkstationId | None = None
    order_id: str | None = None
    product_unit_id: str | None = None
    mold_name: str | None = None
    comb_name: str | None = None
    press_locked: bool = False
    last_transition_at: float = 0.0

    def transition(self, phase: TrayRoutePhase, now: float) -> None:
        allowed = {
            TrayRoutePhase.EMPTY_BUFFER: {
                TrayRoutePhase.CHANGEOVER,
                TrayRoutePhase.MOLD_READY,
            },
            TrayRoutePhase.CHANGEOVER: {TrayRoutePhase.MOLD_READY, TrayRoutePhase.EMPTY_BUFFER},
            TrayRoutePhase.MOLD_READY: {TrayRoutePhase.BASE_READY, TrayRoutePhase.CHANGEOVER},
            TrayRoutePhase.BASE_READY: {TrayRoutePhase.MATERIAL_READY},
            TrayRoutePhase.MATERIAL_READY: {TrayRoutePhase.ASSEMBLY_READY},
            TrayRoutePhase.ASSEMBLY_READY: {TrayRoutePhase.LOCKED},
            TrayRoutePhase.LOCKED: {TrayRoutePhase.OUTFEED},
            TrayRoutePhase.OUTFEED: {TrayRoutePhase.FURNACE},
            TrayRoutePhase.FURNACE: {TrayRoutePhase.FINISHED_GOODS},
            TrayRoutePhase.FINISHED_GOODS: {TrayRoutePhase.RETURNING},
            TrayRoutePhase.RETURNING: {TrayRoutePhase.CHANGEOVER, TrayRoutePhase.EMPTY_BUFFER},
        }
        phase = TrayRoutePhase(phase)
        if phase is not self.phase and phase not in allowed[self.phase]:
            raise ValueError(f"illegal tray transition: {self.phase.value} -> {phase.value}")
        self.phase = phase
        self.last_transition_at = float(now)

    def as_dict(self) -> dict[str, object]:
        return {
            "tray_id": self.tray_id,
            "phase": self.phase.value,
            "owner": self.owner.value,
            "nest_id": None if self.nest_id is None else self.nest_id.value,
            "station_id": None if self.station_id is None else self.station_id.value,
            "order_id": self.order_id,
            "product_unit_id": self.product_unit_id,
            "mold_name": self.mold_name,
            "comb_name": self.comb_name,
            "press_locked": self.press_locked,
            "last_transition_at": self.last_transition_at,
        }


@dataclass(slots=True)
class TurntableState:
    angle_deg: float = 0.0
    target_angle_deg: float = 0.0
    rotating: bool = False
    settled_since: float | None = 0.0
    rotation_started_at: float | None = None
    rotation_duration_s: float = 2.0
    settle_duration_s: float = 0.3
    nest_trays: dict[NestId, str | None] = field(
        default_factory=lambda: {NestId.NEST_A: None, NestId.NEST_B: None}
    )
    blocked_reasons: list[str] = field(default_factory=list)

    def request_rotation(self, now: float, blockers: Iterable[str] = ()) -> bool:
        self.blocked_reasons = [str(value) for value in blockers if str(value)]
        if self.rotating or self.blocked_reasons:
            return False
        if any(tray_id is None for tray_id in self.nest_trays.values()):
            self.blocked_reasons = ["两个转台巢位必须均有已锁定托盘"]
            return False
        self.rotation_started_at = float(now)
        self.target_angle_deg = self.angle_deg + 180.0
        self.rotating = True
        self.settled_since = None
        return True

    def update(self, now: float) -> bool:
        """Advance the S-curve and return ``True`` once a swap has just finished."""

        if not self.rotating or self.rotation_started_at is None:
            return False
        elapsed = max(0.0, float(now) - self.rotation_started_at)
        start = self.target_angle_deg - 180.0
        ratio = elapsed / max(self.rotation_duration_s, 1e-6)
        self.angle_deg = start + 180.0 * quintic_time_scaling(ratio)
        if ratio < 1.0:
            return False
        self.angle_deg = self.target_angle_deg
        self.rotating = False
        self.settled_since = float(now)
        self.nest_trays[NestId.NEST_A], self.nest_trays[NestId.NEST_B] = (
            self.nest_trays[NestId.NEST_B],
            self.nest_trays[NestId.NEST_A],
        )
        return True

    def settled(self, now: float) -> bool:
        return (
            not self.rotating
            and self.settled_since is not None
            and float(now) - self.settled_since >= self.settle_duration_s
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "angle_deg": self.angle_deg % 360.0,
            "target_angle_deg": self.target_angle_deg % 360.0,
            "rotating": self.rotating,
            "settled": self.settled_since is not None,
            "nest_trays": {key.value: value for key, value in self.nest_trays.items()},
            "blocked_reasons": list(self.blocked_reasons),
        }


@dataclass(slots=True)
class ChangeoverStep:
    step_id: str
    action: str
    module_id: str
    source: ModuleLocation
    destination: ModuleLocation
    status: ChangeoverStatus = ChangeoverStatus.PENDING
    started_at: float | None = None
    finished_at: float | None = None
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "module_id": self.module_id,
            "source": self.source.value,
            "destination": self.destination.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "failure_reason": self.failure_reason,
        }


@dataclass(slots=True)
class ChangeoverPlan:
    plan_id: str
    tray_id: str
    comb_module: str
    steps: list[ChangeoverStep]
    current_index: int = 0
    prefetched_modules: list[str] = field(default_factory=list)

    @property
    def current_step(self) -> ChangeoverStep | None:
        if 0 <= self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    @property
    def complete(self) -> bool:
        return bool(self.steps) and all(step.status is ChangeoverStatus.SUCCEEDED for step in self.steps)

    def start_current(self, now: float) -> ChangeoverStep:
        step = self.current_step
        if step is None:
            raise RuntimeError("changeover plan is already complete")
        if step.status is ChangeoverStatus.PENDING:
            step.status = ChangeoverStatus.RUNNING
            step.started_at = float(now)
        return step

    def finish_current(
        self,
        now: float,
        module_locations: dict[str, ModuleLocation],
        *,
        contact_error_m: float,
        speed_m_s: float,
        locked: bool,
    ) -> None:
        step = self.current_step
        if step is None or step.status not in {ChangeoverStatus.RUNNING, ChangeoverStatus.VERIFYING}:
            raise RuntimeError("no active changeover step")
        if module_locations.get(step.module_id) is not step.source:
            raise RuntimeError(f"module ownership mismatch for {step.module_id}")
        if contact_error_m > 0.002 or speed_m_s > 0.01 or not locked:
            step.status = ChangeoverStatus.VERIFYING
            return
        module_locations[step.module_id] = step.destination
        step.status = ChangeoverStatus.SUCCEEDED
        step.finished_at = float(now)
        self.current_index += 1

    def as_dict(self) -> dict[str, object]:
        step = self.current_step
        return {
            "plan_id": self.plan_id,
            "tray_id": self.tray_id,
            "comb_module": self.comb_module,
            "current_step": None if step is None else step.step_id,
            "progress": self.current_index / max(1, len(self.steps)),
            "complete": self.complete,
            "prefetched_modules": list(self.prefetched_modules),
            "steps": [item.as_dict() for item in self.steps],
        }


def build_changeover_plan(plan_id: str, tray_id: str, comb_module: str) -> ChangeoverPlan:
    """Create the visible, ordered changeover sequence for one empty tray."""

    requested = (
        (ModuleKind.MOLD, f"mold_{comb_module}"),
        (ModuleKind.FRONT_COMB, f"front_{comb_module}"),
        (ModuleKind.REAR_COMB, f"rear_{comb_module}"),
        (ModuleKind.FRONT_PRESS, "front_press"),
        (ModuleKind.REAR_PRESS, "rear_press"),
    )
    steps: list[ChangeoverStep] = []
    for index, (kind, module_id) in enumerate(requested, start=1):
        steps.append(
            ChangeoverStep(
                step_id=f"{plan_id}_{index:02d}_{kind.value}",
                action=f"安装{kind.value.lower()}",
                module_id=module_id,
                source=ModuleLocation.RACK,
                destination=ModuleLocation.TRAY,
            )
        )
    return ChangeoverPlan(plan_id=plan_id, tray_id=tray_id, comb_module=comb_module, steps=steps)


def validate_cell_state(trays: Iterable[TrayRouteState], turntable: TurntableState) -> None:
    tray_list = list(trays)
    tray_ids = [tray.tray_id for tray in tray_list]
    if len(tray_ids) != len(set(tray_ids)):
        raise ValueError("tray_id must be unique")
    nested = [value for value in turntable.nest_trays.values() if value is not None]
    if len(nested) != len(set(nested)):
        raise ValueError("one tray cannot occupy both turntable nests")
    unknown = set(nested).difference(tray_ids)
    if unknown:
        raise ValueError(f"turntable contains unknown trays: {sorted(unknown)}")
    for value in (turntable.rotation_duration_s, turntable.settle_duration_s):
        if not isfinite(value) or value < 0:
            raise ValueError("turntable durations must be finite and non-negative")


def validate_async_line_state(
    trays: Iterable[TrayRouteState],
    workstations: Iterable[WorkstationState],
    transfers: Iterable[AsyncTransferState],
    *,
    wip_limit: int = 3,
) -> None:
    """Validate unique physical ownership across every station and slide."""

    tray_list = list(trays)
    tray_ids = [tray.tray_id for tray in tray_list]
    if len(tray_ids) != len(set(tray_ids)):
        raise ValueError("tray_id must be unique")
    if int(wip_limit) < 1 or int(wip_limit) > len(tray_ids):
        raise ValueError("WIP limit must fit the available pallet pool")
    occupied = [station.tray_id for station in workstations if station.tray_id is not None]
    occupied.extend(transfer.tray_id for transfer in transfers if transfer.tray_id is not None)
    if len(occupied) != len(set(occupied)):
        raise ValueError("one pallet cannot belong to two stations/transfers")
    unknown = set(occupied).difference(tray_ids)
    if unknown:
        raise ValueError(f"asynchronous line contains unknown pallets: {sorted(unknown)}")
    active_wip = sum(
        tray.owner
        not in {
            TrayOwner.EMPTY_BUFFER,
            TrayOwner.OUTPUT,
        }
        for tray in tray_list
    )
    if active_wip > int(wip_limit):
        raise ValueError(f"active WIP {active_wip} exceeds limit {wip_limit}")


__all__ = [
    "AsyncTransferState",
    "ChangeoverPlan",
    "ChangeoverStatus",
    "ChangeoverStep",
    "ModuleKind",
    "ModuleLocation",
    "NestId",
    "TransferId",
    "TransferStatus",
    "TrayOwner",
    "TrayRoutePhase",
    "TrayRouteState",
    "TurntableState",
    "WorkstationId",
    "WorkstationState",
    "build_changeover_plan",
    "validate_async_line_state",
    "validate_cell_state",
]
