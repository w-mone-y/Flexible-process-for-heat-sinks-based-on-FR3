"""Capability ontology and product routing models.

These types back step A of the flexibility upgrade: a process is described by
*data* (which capability an operation needs, which parameters it carries) rather
than by a hand-written DAG.  Three artefacts cooperate:

``config/capabilities.yaml``
    Factory-level contract per capability: required tool class, parameter
    schema, parametric duration model, zones, pre/post conditions.

``config/routings/<product>.yaml``
    Per-product operation list.  Each operation names a capability, its
    predecessors and its parameters, and may declare ``alternatives`` — the OR
    branches that make process flexibility explicit.

``config/resources.yaml``
    Per-resource capability declarations with speed factors, parameter limits
    and fixture compatibility, so eligibility can be derived instead of
    hard-coded.

Nothing here touches MuJoCo, so the whole layer stays unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .duration_model import DurationModel


class ParamType(str, Enum):
    """Parameter primitive types supported by a capability schema."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"


def is_placeholder(value: Any) -> bool:
    """True for a ``$name`` routing placeholder resolved from the process plan.

    Routings stay product-agnostic by writing ``path_count: $path_count`` instead
    of a literal.  Such values pass schema validation at load time and are fully
    validated once the compiler substitutes the plan's actual value.
    """

    return isinstance(value, str) and len(value) > 1 and value.startswith("$")


def placeholder_name(value: str) -> str:
    """Strip the leading ``$`` from a placeholder token."""

    return str(value)[1:]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One declared operation parameter with an optional inclusive range."""

    name: str
    type: ParamType
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    default: Any = None
    required: bool = True

    def check(self, value: Any) -> tuple[bool, str]:
        """Validate one value, returning ``(ok, reason)`` with a Chinese reason."""

        if self.type is ParamType.BOOL:
            if not isinstance(value, bool):
                return False, f"参数 {self.name} 必须是布尔值，实际 {type(value).__name__}"
            return True, ""
        if self.type is ParamType.STRING:
            if not isinstance(value, str):
                return False, f"参数 {self.name} 必须是字符串，实际 {type(value).__name__}"
            if self.choices and value not in self.choices:
                return False, f"参数 {self.name} 必须是 {list(self.choices)} 之一，实际 {value!r}"
            return True, ""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, f"参数 {self.name} 必须是数值，实际 {type(value).__name__}"
        if self.type is ParamType.INT and not float(value).is_integer():
            return False, f"参数 {self.name} 必须是整数，实际 {value}"
        number = float(value)
        if not isfinite(number):
            return False, f"参数 {self.name} 必须是有限数值"
        if self.minimum is not None and number < self.minimum:
            return False, f"参数 {self.name} 必须不小于 {self.minimum:g}，实际 {number:g}"
        if self.maximum is not None and number > self.maximum:
            return False, f"参数 {self.name} 必须不大于 {self.maximum:g}，实际 {number:g}"
        return True, ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "min": self.minimum,
            "max": self.maximum,
            "choices": list(self.choices),
            "default": self.default,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """Factory-level contract for one kind of process operation."""

    name: str
    task_type: str
    requires_tool_class: str | None
    param_schema: tuple[ParamSpec, ...]
    duration_model: DurationModel
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    zones: tuple[str, ...] = ()
    preemptive: bool = True
    description_zh: str = ""
    source_file: Path | None = None

    @property
    def param_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.param_schema)

    def spec_for(self, name: str) -> ParamSpec | None:
        for item in self.param_schema:
            if item.name == name:
                return item
        return None

    def normalize_params(
        self,
        params: Mapping[str, Any],
        *,
        allow_placeholders: bool = False,
    ) -> tuple[dict[str, Any], str]:
        """Fill defaults and validate; returns ``(params, error)``.

        ``error`` is an empty string on success.  Unknown keys are rejected so a
        typo in a routing file never becomes a silently ignored parameter.

        With ``allow_placeholders`` (routing load time) a ``$name`` token is
        accepted verbatim and re-validated after the compiler substitutes the
        plan's real value.
        """

        unknown = sorted(set(params) - self.param_names)
        if unknown:
            return {}, f"能力 {self.name} 不接受参数 {unknown}，已声明参数为 {sorted(self.param_names)}"
        resolved: dict[str, Any] = {}
        for spec in self.param_schema:
            if spec.name in params:
                value = params[spec.name]
            elif spec.default is not None:
                value = spec.default
            elif spec.required:
                return {}, f"能力 {self.name} 缺少必填参数 {spec.name}"
            else:
                continue
            if allow_placeholders and is_placeholder(value):
                resolved[spec.name] = value
                continue
            ok, reason = spec.check(value)
            if not ok:
                return {}, reason
            resolved[spec.name] = int(value) if spec.type is ParamType.INT else value
        return resolved, ""

    def duration_for(self, params: Mapping[str, Any]) -> float:
        """Parametric nominal duration before any resource speed factor."""

        return self.duration_model.evaluate(params)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "requires_tool_class": self.requires_tool_class,
            "param_schema": [item.as_dict() for item in self.param_schema],
            "duration_model": self.duration_model.expression,
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "zones": list(self.zones),
            "preemptive": self.preemptive,
            "description_zh": self.description_zh,
        }


@dataclass(frozen=True, slots=True)
class OperationAlternative:
    """One OR branch of an operation: a different way to achieve the same effect."""

    mode: str
    capability: str
    cost_hint: float = 1.0
    params: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "capability": self.capability,
            "cost_hint": self.cost_hint,
            "params": dict(self.params),
        }


@dataclass(frozen=True, slots=True)
class BatchPolicy:
    """Declarative batching rule for furnace-style grouped operations."""

    group_key: str
    max_units: int

    def as_dict(self) -> dict[str, Any]:
        return {"group_key": self.group_key, "max_units": self.max_units}


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One routing step, possibly with OR alternatives and a batching policy."""

    operation_id: str
    capability: str
    after: tuple[str, ...] = ()
    # Per-fin operations only: depend on the *previous* fin's node of another
    # operation.  This expresses single-gripper reality ("fin i cannot be picked
    # until fin i-1 has been installed") as data instead of as builder code.
    after_previous: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    alternatives: tuple[OperationAlternative, ...] = ()
    batchable: BatchPolicy | None = None
    station: str | None = None
    per_unit_of: str | None = None
    retry_limit: int = 0
    description_zh: str = ""

    @property
    def has_alternatives(self) -> bool:
        return len(self.alternatives) > 1

    def capability_choices(self) -> tuple[str, ...]:
        """Every capability that can realize this operation, primary first."""

        if not self.alternatives:
            return (self.capability,)
        return tuple(dict.fromkeys(item.capability for item in self.alternatives))

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "capability": self.capability,
            "after": list(self.after),
            "after_previous": list(self.after_previous),
            "params": dict(self.params),
            "alternatives": [item.as_dict() for item in self.alternatives],
            "batchable": None if self.batchable is None else self.batchable.as_dict(),
            "station": self.station,
            "per_unit_of": self.per_unit_of,
            "retry_limit": self.retry_limit,
            "description_zh": self.description_zh,
        }


@dataclass(frozen=True, slots=True)
class RoutingSpec:
    """A product's full operation list — the data that replaces a hand-written DAG."""

    schema_version: int
    routing_id: str
    product: str
    operations: tuple[OperationSpec, ...]
    description_zh: str = ""
    source_file: Path | None = None

    def operation(self, operation_id: str) -> OperationSpec:
        for item in self.operations:
            if item.operation_id == operation_id:
                return item
        raise KeyError(f"unknown operation: {operation_id}")

    @property
    def operation_ids(self) -> tuple[str, ...]:
        return tuple(item.operation_id for item in self.operations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "routing_id": self.routing_id,
            "product": self.product,
            "description_zh": self.description_zh,
            "operations": [item.as_dict() for item in self.operations],
        }


@dataclass(frozen=True, slots=True)
class ResourceCapability:
    """What one resource can do, how fast, and within which parameter window."""

    name: str
    speed_factor: float = 1.0
    preemptive: bool = True
    param_limits: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def accepts(self, params: Mapping[str, Any]) -> tuple[bool, str]:
        """Check operation parameters against this resource's physical window."""

        for key, (low, high) in self.param_limits.items():
            if key not in params:
                continue
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if number < low - 1e-12 or number > high + 1e-12:
                return False, f"参数 {key}={number:g} 超出该资源许可范围 [{low:g}, {high:g}]"
        return True, ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "speed_factor": self.speed_factor,
            "preemptive": self.preemptive,
            "param_limits": {key: list(value) for key, value in self.param_limits.items()},
        }


@dataclass(frozen=True, slots=True)
class CapabilityCatalog:
    """Immutable capability registry with a tool-class index."""

    capabilities: Mapping[str, CapabilitySpec]
    source_file: Path | None = None

    def get(self, name: str) -> CapabilitySpec:
        try:
            return self.capabilities[str(name).upper()]
        except KeyError as exc:
            raise KeyError(f"unknown capability: {name}") from exc

    def __contains__(self, name: object) -> bool:
        return str(name).upper() in self.capabilities

    def __iter__(self):
        return iter(self.capabilities.values())

    def __len__(self) -> int:
        return len(self.capabilities)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.capabilities))

    def as_dict(self) -> dict[str, Any]:
        return {name: spec.as_dict() for name, spec in sorted(self.capabilities.items())}


__all__ = [
    "BatchPolicy",
    "CapabilityCatalog",
    "CapabilitySpec",
    "OperationAlternative",
    "OperationSpec",
    "ParamSpec",
    "ParamType",
    "ResourceCapability",
    "RoutingSpec",
    "is_placeholder",
    "placeholder_name",
]
