"""Serializable contracts shared by reference solvers and validators."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from typing import Any, Mapping


class PlanStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True, slots=True)
class PlanOperation:
    task_id: str
    resource_id: str
    start_s: float
    end_s: float
    batch_id: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.resource_id:
            raise ValueError("plan operation requires task_id and resource_id")
        values = (float(self.start_s), float(self.end_s))
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("plan operation times must be finite and non-negative")
        if self.end_s < self.start_s:
            raise ValueError("plan operation end must not precede start")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "batch_id": self.batch_id,
        }


@dataclass(frozen=True, slots=True)
class PlanViolation:
    code: str
    message: str
    task_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "task_ids": list(self.task_ids)}


@dataclass(frozen=True, slots=True)
class PlanValidation:
    valid: bool
    violations: tuple[PlanViolation, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": [violation.as_dict() for violation in self.violations],
        }


@dataclass(frozen=True, slots=True)
class ReferencePlan:
    status: PlanStatus | str
    operations: tuple[PlanOperation, ...] = ()
    makespan_s: float = 0.0
    weighted_tardiness_s: float = 0.0
    objective_value: float = 0.0
    best_bound: float | None = None
    optimality_gap: float | None = None
    solve_time_s: float = 0.0
    solver_name: str = "CP-SAT"
    plan_version: int = 0
    snapshot_fingerprint: str = ""
    timed_out: bool = False
    message: str = ""
    validation: PlanValidation | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PlanStatus(self.status))
        values = (self.makespan_s, self.weighted_tardiness_s, self.objective_value, self.solve_time_s)
        if any(not isfinite(float(value)) or value < 0.0 for value in values):
            raise ValueError("reference-plan metrics must be finite and non-negative")
        if self.best_bound is not None and not isfinite(float(self.best_bound)):
            raise ValueError("best_bound must be finite")
        if self.optimality_gap is not None and (
            not isfinite(float(self.optimality_gap)) or self.optimality_gap < 0.0
        ):
            raise ValueError("optimality_gap must be finite and non-negative")

    def with_validation(self, validation: PlanValidation) -> "ReferencePlan":
        return replace(self, validation=validation)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "operations": [operation.as_dict() for operation in self.operations],
            "makespan_s": round(self.makespan_s, 6),
            "weighted_tardiness_s": round(self.weighted_tardiness_s, 6),
            "objective_value": round(self.objective_value, 6),
            "best_bound": None if self.best_bound is None else round(self.best_bound, 6),
            "optimality_gap": (
                None if self.optimality_gap is None else round(self.optimality_gap, 8)
            ),
            "solve_time_s": round(self.solve_time_s, 6),
            "solver_name": self.solver_name,
            "plan_version": self.plan_version,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "timed_out": self.timed_out,
            "message": self.message,
            "validation": None if self.validation is None else self.validation.as_dict(),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "PlanOperation",
    "PlanStatus",
    "PlanValidation",
    "PlanViolation",
    "ReferencePlan",
]
