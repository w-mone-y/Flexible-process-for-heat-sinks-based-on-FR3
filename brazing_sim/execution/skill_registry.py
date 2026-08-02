"""Skill registry and adapters for existing non-blocking actors."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Callable, Protocol

from ..domain import Actor, TaskSpec, TaskType as LegacyTaskType
from ..planning.task_models import ManufacturingTask, TaskType


@dataclass(frozen=True, slots=True)
class PhysicalCompletionEvidence:
    """Measured proof that a physical skill has reached its terminal state."""

    observed_at: float
    source: str
    checks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isfinite(float(self.observed_at)) or float(self.observed_at) < 0.0:
            raise ValueError("physical completion time must be finite and non-negative")
        if not str(self.source).strip():
            raise ValueError("physical completion source must not be empty")
        normalized = tuple(str(check).strip() for check in self.checks if str(check).strip())
        if not normalized:
            raise ValueError("physical completion must contain at least one measured check")
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(self, "checks", normalized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_at": float(self.observed_at),
            "source": self.source,
            "checks": list(self.checks),
        }


@dataclass(frozen=True, slots=True)
class SkillExecutionResult:
    running: bool = False
    succeeded: bool = False
    failed: bool = False
    failure_code: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    completion_evidence: PhysicalCompletionEvidence | None = None

    def __post_init__(self) -> None:
        terminal = int(self.succeeded) + int(self.failed)
        if terminal > 1 or (self.running and terminal):
            raise ValueError("skill result flags are mutually exclusive")
        if self.completion_evidence is not None and not self.succeeded:
            raise ValueError("completion evidence is only valid for a successful result")

    @classmethod
    def running_result(cls, metrics: dict[str, Any] | None = None) -> "SkillExecutionResult":
        return cls(running=True, metrics=dict(metrics or {}))

    @classmethod
    def success(
        cls,
        metrics: dict[str, Any] | None = None,
        *,
        completion_evidence: PhysicalCompletionEvidence | None = None,
    ) -> "SkillExecutionResult":
        return cls(
            succeeded=True,
            metrics=dict(metrics or {}),
            completion_evidence=completion_evidence,
        )

    @classmethod
    def failure(cls, code: str, metrics: dict[str, Any] | None = None) -> "SkillExecutionResult":
        return cls(failed=True, failure_code=str(code), metrics=dict(metrics or {}))


class Skill(Protocol):
    def start(self, task: ManufacturingTask, resource_id: str, context: Any, now: float) -> None: ...

    def update(self, now: float, dt: float) -> SkillExecutionResult: ...

    def cancel(self, task_id: str) -> None: ...


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[TaskType, Skill] = {}
        self._factories: dict[TaskType, Callable[[], Skill]] = {}
        self._physical_evidence_required: set[TaskType] = set()

    def register(self, task_type: TaskType | str, skill: Skill, *, replace: bool = False) -> None:
        key = TaskType(task_type)
        if (key in self._skills or key in self._factories) and not replace:
            raise ValueError(f"skill already registered for {key.value}")
        self._factories.pop(key, None)
        self._physical_evidence_required.discard(key)
        self._skills[key] = skill

    def register_factory(
        self,
        task_type: TaskType | str,
        factory: Callable[[], Skill],
        *,
        replace: bool = False,
        requires_physical_evidence: bool = False,
    ) -> None:
        """Register a per-execution skill factory.

        Factories are required when one task type can run concurrently on two
        capable resources (for example V2 fin installation on Arm1 and Arm3).
        """

        key = TaskType(task_type)
        if (key in self._skills or key in self._factories) and not replace:
            raise ValueError(f"skill already registered for {key.value}")
        if not callable(factory):
            raise TypeError("skill factory must be callable")
        self._skills.pop(key, None)
        self._factories[key] = factory
        if requires_physical_evidence:
            self._physical_evidence_required.add(key)
        else:
            self._physical_evidence_required.discard(key)

    def get(self, task_type: TaskType | str) -> Skill:
        key = TaskType(task_type)
        if key in self._factories:
            return self._factories[key]()
        try:
            return self._skills[key]
        except KeyError as exc:
            raise KeyError(f"no skill registered for {key.value}") from exc

    def supports(self, task_type: TaskType | str) -> bool:
        key = TaskType(task_type)
        return key in self._skills or key in self._factories

    def create(self, task_type: TaskType | str) -> Skill:
        """Create or obtain the skill instance for one concrete execution."""

        return self.get(task_type)

    def requires_physical_evidence(self, task_type: TaskType | str) -> bool:
        return TaskType(task_type) in self._physical_evidence_required

    def snapshot(self) -> dict[str, str]:
        snapshot = {key.value: type(skill).__name__ for key, skill in self._skills.items()}
        snapshot.update(
            {
                key.value: f"factory:{getattr(factory, '__name__', type(factory).__name__)}"
                for key, factory in self._factories.items()
            }
        )
        return dict(sorted(snapshot.items()))


class TimedSkill:
    """Deterministic tick-driven skill used by experiments and tests."""

    def __init__(self, duration: float | None = None) -> None:
        self.duration = duration
        self.task: ManufacturingTask | None = None
        self.started_at = 0.0
        self.finished_at = 0.0
        self.cancelled = False
        self.failure_code: str | None = None

    def start(self, task: ManufacturingTask, resource_id: str, context: Any, now: float) -> None:
        del resource_id, context
        if self.task is not None:
            raise RuntimeError("timed skill is already busy")
        self.task = task
        self.started_at = float(now)
        self.finished_at = float(now) + (
            task.estimated_duration if self.duration is None else float(self.duration)
        )
        self.cancelled = False
        self.failure_code = task.payload.get("forced_failure_code")

    def update(self, now: float, dt: float) -> SkillExecutionResult:
        del dt
        if self.task is None:
            return SkillExecutionResult.success()
        if self.cancelled:
            self.task = None
            return SkillExecutionResult.failure("CANCELLED")
        if float(now) < self.finished_at:
            progress = (float(now) - self.started_at) / max(1e-9, self.finished_at - self.started_at)
            return SkillExecutionResult.running_result({"progress": max(0.0, min(1.0, progress))})
        failure_code = self.failure_code
        elapsed = max(0.0, float(now) - self.started_at)
        self.task = None
        if failure_code:
            return SkillExecutionResult.failure(failure_code, {"elapsed": elapsed})
        return SkillExecutionResult.success({"elapsed": elapsed})

    def cancel(self, task_id: str) -> None:
        if self.task is not None and self.task.task_id == task_id:
            self.cancelled = True
            self.task = None


LEGACY_TASK_MAPPING: dict[TaskType, LegacyTaskType | None] = {
    TaskType.PICK_BASE_PLATE: None,
    TaskType.PLACE_BASE_PLATE: LegacyTaskType.LOAD_BASE,
    TaskType.PREPARE_FIN_TOOL: LegacyTaskType.PREPARE_FIN_TOOL,
    TaskType.DISPENSE_BRAZING: LegacyTaskType.APPLY_MATERIAL,
    TaskType.INSPECT_BRAZING: LegacyTaskType.MATERIAL_INSPECT,
    TaskType.REWORK_BRAZING: LegacyTaskType.REAPPLY_MATERIAL,
    TaskType.CONFIGURE_COMB: LegacyTaskType.CONFIGURE_COMB,
    TaskType.PICK_FIN: None,
    TaskType.INSTALL_FIN: LegacyTaskType.INSERT_FIN,
    TaskType.INSPECT_FINS: LegacyTaskType.PRE_INSPECT,
    TaskType.REINSTALL_FIN: LegacyTaskType.ADJUST_FIN,
    TaskType.APPLY_PRESS: LegacyTaskType.PRESS_FIXTURE,
    TaskType.LOCK_FIXTURE: LegacyTaskType.LOCK_FIXTURE,
    TaskType.POST_BRAZE_INSPECTION: LegacyTaskType.POST_INSPECT,
}


def _legacy_actor(resource_id: str) -> Actor | str:
    mapping = {
        "ARM1": Actor.ARM1,
        "ARM2": Actor.ARM2,
        "ARM3": Actor.ARM3,
        "FIXTURE": Actor.FIXTURE,
        "FURNACE": Actor.FURNACE,
        "OUTFEED": Actor.CONVEYOR,
        "ELEVATOR": "lift_transfer",
        "TRANSFER_FORK": "rack_pusher",
    }
    return mapping.get(str(resource_id).upper(), str(resource_id).lower())


class ActorSkillAdapter:
    """Translate a V2 task to the existing ``start_task/poll_task`` actor API."""

    def __init__(self, actor: Any, legacy_task_type: LegacyTaskType | None = None) -> None:
        self.actor = actor
        self.legacy_task_type = legacy_task_type
        self.task: ManufacturingTask | None = None
        self.command: TaskSpec | None = None

    def start(self, task: ManufacturingTask, resource_id: str, context: Any, now: float) -> None:
        del context
        if self.task is not None:
            raise RuntimeError("actor skill is already busy")
        task_type = self.legacy_task_type
        if task_type is None:
            task_type = LEGACY_TASK_MAPPING.get(task.task_type)
        self.task = task
        if task_type is None:
            self.command = None
            return
        self.command = TaskSpec(
            task_id=task.task_id,
            actor=_legacy_actor(resource_id),
            task_type=task_type,
            payload=dict(task.payload),
            timeout=max(1.0, task.estimated_duration * 5.0),
            max_retries=task.retry_limit,
            retries=task.retry_count,
        )
        self.actor.start_task(self.command, now)

    def update(self, now: float, dt: float) -> SkillExecutionResult:
        del dt
        if self.task is None:
            return SkillExecutionResult.success()
        if self.command is None:
            self.task = None
            return SkillExecutionResult.success({"compatibility_milestone": True})
        try:
            result = self.actor.poll_task(now)
        except Exception as exc:
            self.task = None
            self.command = None
            return SkillExecutionResult.failure("ACTOR_EXCEPTION", {"error": str(exc)})
        value = str(getattr(result, "value", result)).upper()
        if result is False or value in {"FAILED", "ERROR", "CANCELLED"}:
            code = str(getattr(self.actor, "error", "ACTOR_FAILED") or "ACTOR_FAILED")
            self.task = None
            self.command = None
            return SkillExecutionResult.failure(code)
        if result is True or value in {"SUCCEEDED", "SUCCESS", "DONE", "COMPLETE", "COMPLETED"}:
            payload = dict(self.command.payload)
            self.task = None
            self.command = None
            return SkillExecutionResult.success(payload)
        return SkillExecutionResult.running_result()

    def cancel(self, task_id: str) -> None:
        if self.task is None or self.task.task_id != task_id:
            return
        if hasattr(self.actor, "cancel"):
            self.actor.cancel()
        self.task = None
        self.command = None


__all__ = [
    "ActorSkillAdapter",
    "LEGACY_TASK_MAPPING",
    "PhysicalCompletionEvidence",
    "Skill",
    "SkillExecutionResult",
    "SkillRegistry",
    "TimedSkill",
]
