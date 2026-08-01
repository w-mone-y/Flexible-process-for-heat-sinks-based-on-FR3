"""Typed fault and recovery-plan models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FaultType(str, Enum):
    BRAZING_MISSING = "BRAZING_MISSING"
    BRAZING_PATH_DEVIATION = "BRAZING_PATH_DEVIATION"
    FIN_PICK_FAILED = "FIN_PICK_FAILED"
    FIN_GEOMETRY_FAILED = "FIN_GEOMETRY_FAILED"
    ARM_UNAVAILABLE = "ARM_UNAVAILABLE"
    RACK_LAYER_UNAVAILABLE = "RACK_LAYER_UNAVAILABLE"
    ELEVATOR_TIMEOUT = "ELEVATOR_TIMEOUT"
    FORK_TIMEOUT = "FORK_TIMEOUT"
    FURNACE_DOOR_INTERLOCK = "FURNACE_DOOR_INTERLOCK"
    FURNACE_PROFILE = "FURNACE_PROFILE"
    CONTACT_SAFETY_STOP = "CONTACT_SAFETY_STOP"
    TRAY_STATE_INCONSISTENT = "TRAY_STATE_INCONSISTENT"


class RecoveryStatus(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(slots=True)
class FaultRecord:
    fault_id: str
    fault_type: FaultType | str
    source: str
    related_task_id: str | None
    sim_time: float
    recoverable: bool
    details: dict[str, Any] = field(default_factory=dict)
    recovered: bool = False
    recovery_id: str | None = None

    def __post_init__(self) -> None:
        self.fault_type = FaultType(self.fault_type)
        if not self.fault_id or not self.source:
            raise ValueError("fault_id and source must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "source": self.source,
            "related_task_id": self.related_task_id,
            "sim_time": self.sim_time,
            "recoverable": self.recoverable,
            "details": dict(self.details),
            "recovered": self.recovered,
            "recovery_id": self.recovery_id,
        }


@dataclass(slots=True)
class RecoveryStep:
    step_id: str
    description: str
    task_id: str | None = None
    status: str = "PENDING"

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "task_id": self.task_id,
            "status": self.status,
        }


@dataclass(slots=True)
class RecoveryPlan:
    recovery_id: str
    fault_id: str
    strategy: str
    steps: list[RecoveryStep] = field(default_factory=list)
    status: RecoveryStatus | str = RecoveryStatus.PLANNED
    retry_count: int = 0
    retry_limit: int = 0
    created_at: float = 0.0
    completed_at: float | None = None
    message: str = ""

    def __post_init__(self) -> None:
        self.status = RecoveryStatus(self.status)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recovery_id": self.recovery_id,
            "fault_id": self.fault_id,
            "strategy": self.strategy,
            "steps": [step.as_dict() for step in self.steps],
            "status": self.status.value,
            "retry_count": self.retry_count,
            "retry_limit": self.retry_limit,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "message": self.message,
        }


__all__ = ["FaultRecord", "FaultType", "RecoveryPlan", "RecoveryStatus", "RecoveryStep"]
