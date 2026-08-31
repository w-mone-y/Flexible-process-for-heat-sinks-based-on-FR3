"""Immutable digital-twin snapshots and typed decision events.

This module is deliberately independent of MuJoCo and of either runtime's
mutable state. It provides a stable boundary for shadow scheduling,
experiments and UI consumers without introducing a second source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from .events import EventType, SystemEvent


def _freeze(value: Any) -> Any:
    """Recursively copy JSON-like state into immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, Enum):
        return _freeze(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"digital-twin state must be JSON-like, got {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _records(state: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value = state.get(key, ())
    if isinstance(value, Mapping):
        value = tuple(value.values())
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


@dataclass(frozen=True, slots=True)
class DigitalTwinSnapshot:
    """A deeply immutable view of one runtime state at a decision boundary."""

    state: Mapping[str, Any]
    source_name: str
    captured_at: float
    plan_version: int = 0
    schema_version: int = 1
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not str(self.source_name).strip():
            raise ValueError("source_name must not be empty")
        if not isfinite(float(self.captured_at)) or self.captured_at < 0.0:
            raise ValueError("captured_at must be finite and non-negative")
        if int(self.plan_version) < 0:
            raise ValueError("plan_version must be non-negative")
        frozen = _freeze(self.state)
        if not isinstance(frozen, Mapping):
            raise TypeError("digital-twin state must be a mapping")
        object.__setattr__(self, "state", frozen)
        canonical = json.dumps(
            _thaw(frozen),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        ).encode("utf-8")
        object.__setattr__(self, "fingerprint", hashlib.sha256(canonical).hexdigest())

    @classmethod
    def from_mapping(
        cls,
        state: Mapping[str, Any],
        *,
        source_name: str,
        captured_at: float | None = None,
        plan_version: int | None = None,
    ) -> "DigitalTwinSnapshot":
        sim_time = float(state.get("sim_time", 0.0))
        return cls(
            state=state,
            source_name=source_name,
            captured_at=sim_time if captured_at is None else float(captured_at),
            plan_version=0 if plan_version is None else int(plan_version),
            schema_version=int(state.get("schema_version", 1)),
        )

    @property
    def sim_time(self) -> float:
        return float(self.state.get("sim_time", self.captured_at))

    @property
    def orders(self) -> tuple[Mapping[str, Any], ...]:
        return _records(self.state, "orders")

    @property
    def tasks(self) -> tuple[Mapping[str, Any], ...]:
        return _records(self.state, "tasks")

    @property
    def units(self) -> tuple[Mapping[str, Any], ...]:
        return _records(self.state, "units")

    @property
    def ready_task_ids(self) -> tuple[str, ...]:
        return tuple(
            str(task["task_id"])
            for task in self.tasks
            if task.get("status") == "READY" and task.get("task_id") is not None
        )

    @property
    def running_task_ids(self) -> tuple[str, ...]:
        return tuple(
            str(task["task_id"])
            for task in self.tasks
            if task.get("status") == "RUNNING" and task.get("task_id") is not None
        )

    @property
    def active_resources(self) -> tuple[str, ...]:
        resources = self.state.get("resources_v2", {})
        if not isinstance(resources, Mapping):
            return ()
        return tuple(
            sorted(
                str(resource_id)
                for resource_id, resource in resources.items()
                if isinstance(resource, Mapping)
                and str(resource.get("status", "")).upper() in {"BUSY", "RESERVED", "RECOVERING"}
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable export without exposing the internal state."""

        return _thaw(self.state)


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    """Typed, serialisable event emitted by a decision or safety boundary."""

    event_type: EventType | str
    sim_time: float
    source: str
    plan_version: int = 0
    trigger: str = ""
    task_ids: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType(self.event_type))
        if not isfinite(float(self.sim_time)) or self.sim_time < 0.0:
            raise ValueError("sim_time must be finite and non-negative")
        if int(self.plan_version) < 0:
            raise ValueError("plan_version must be non-negative")
        object.__setattr__(self, "task_ids", tuple(str(task_id) for task_id in self.task_ids))
        frozen_payload = _freeze(self.payload)
        if not isinstance(frozen_payload, Mapping):
            raise TypeError("decision event payload must be a mapping")
        object.__setattr__(self, "payload", frozen_payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "sim_time": float(self.sim_time),
            "source": self.source,
            "plan_version": int(self.plan_version),
            "trigger": self.trigger,
            "task_ids": list(self.task_ids),
            "payload": _thaw(self.payload),
        }

    def as_system_event(self) -> SystemEvent:
        encoded = self.as_dict()
        return SystemEvent(
            event_type=self.event_type,
            sim_time=self.sim_time,
            source=self.source,
            payload={
                "plan_version": encoded["plan_version"],
                "trigger": encoded["trigger"],
                "task_ids": encoded["task_ids"],
                **encoded["payload"],
            },
        )


__all__ = ["DecisionEvent", "DigitalTwinSnapshot"]
