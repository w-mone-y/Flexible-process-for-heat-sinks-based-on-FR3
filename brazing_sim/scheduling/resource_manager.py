"""Capability-aware resource state and atomic reservations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Iterable, Mapping

from ..flexible.capability_models import ResourceCapability


class ResourceStatus(str, Enum):
    IDLE = "IDLE"
    RESERVED = "RESERVED"
    BUSY = "BUSY"
    FAULTED = "FAULTED"
    OFFLINE = "OFFLINE"
    RECOVERING = "RECOVERING"


@dataclass(slots=True)
class ResourceState:
    resource_id: str
    resource_type: str
    status: ResourceStatus | str = ResourceStatus.IDLE
    current_task_id: str | None = None
    current_tool: str | None = None
    available_tools: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)
    occupied_zones: set[str] = field(default_factory=set)
    estimated_available_time: float = 0.0
    fault_code: str | None = None
    reserved_at: float | None = None
    busy_since: float | None = None
    last_state_change: float = 0.0
    # Step A/B additions.  ``capabilities`` keeps its legacy meaning (a TaskType
    # allow-list) so V1/V2 dispatch is unchanged; the two fields below carry the
    # capability-ontology view used for delayed binding.
    process_capabilities: dict[str, ResourceCapability] = field(default_factory=dict)
    tool_classes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.resource_id = str(self.resource_id).upper()
        self.resource_type = str(self.resource_type).upper()
        self.status = ResourceStatus(self.status)
        self.available_tools = {str(item) for item in self.available_tools}
        self.capabilities = {str(item).upper() for item in self.capabilities}
        self.occupied_zones = {str(item).upper() for item in self.occupied_zones}
        self.process_capabilities = {
            str(key).upper(): value for key, value in dict(self.process_capabilities).items()
        }
        self.tool_classes = {str(key): str(value).upper() for key, value in dict(self.tool_classes).items()}

    @property
    def available(self) -> bool:
        return self.status is ResourceStatus.IDLE

    def supports(self, task_type: str, required_tool: str | None = None) -> bool:
        if self.capabilities and str(task_type).upper() not in self.capabilities:
            return False
        if required_tool is None:
            return True
        return required_tool == self.current_tool or required_tool in self.available_tools

    # ------------------------------------------------------------------ step B
    def has_capability(self, capability: str) -> bool:
        """True when this resource declares the named process capability."""

        return str(capability).upper() in self.process_capabilities

    def capability(self, capability: str) -> ResourceCapability | None:
        return self.process_capabilities.get(str(capability).upper())

    def tools_for_class(self, tool_class: str) -> tuple[str, ...]:
        """Physical tools of this resource belonging to ``tool_class``."""

        wanted = str(tool_class).upper()
        return tuple(sorted(tool for tool, value in self.tool_classes.items() if value == wanted))

    def provides_tool_class(self, tool_class: str | None) -> bool:
        """True when a tool of the requested class is mounted or mountable."""

        if tool_class is None:
            return True
        candidates = self.tools_for_class(tool_class)
        if not candidates:
            return False
        return any(tool == self.current_tool or tool in self.available_tools for tool in candidates)

    def speed_factor_for(self, capability: str) -> float:
        spec = self.capability(capability)
        return 1.0 if spec is None else float(spec.speed_factor)

    def accepts_capability(
        self,
        capability: str,
        params: Mapping[str, Any] | None = None,
        *,
        tool_class: str | None = None,
    ) -> tuple[bool, str]:
        """Capability-level eligibility check used for delayed binding.

        Returns ``(ok, reason)`` where ``reason`` is a Chinese explanation the UI
        can show verbatim when a candidate is rejected.
        """

        name = str(capability).upper()
        spec = self.process_capabilities.get(name)
        if spec is None:
            return False, f"资源 {self.resource_id} 未声明能力 {name}"
        if not self.provides_tool_class(tool_class):
            return False, f"资源 {self.resource_id} 没有 {tool_class} 类工具"
        if params:
            ok, reason = spec.accepts(params)
            if not ok:
                return False, f"资源 {self.resource_id}：{reason}"
        return True, ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "current_tool": self.current_tool,
            "available_tools": sorted(self.available_tools),
            "capabilities": sorted(self.capabilities),
            "process_capabilities": {
                key: value.as_dict() for key, value in sorted(self.process_capabilities.items())
            },
            "tool_classes": dict(sorted(self.tool_classes.items())),
            "occupied_zones": sorted(self.occupied_zones),
            "estimated_available_time": self.estimated_available_time,
            "fault_code": self.fault_code,
            "reserved_at": self.reserved_at,
            "busy_since": self.busy_since,
            "last_state_change": self.last_state_change,
        }


class ResourceManager:
    def __init__(self, resources: Iterable[ResourceState] = ()) -> None:
        self._resources: dict[str, ResourceState] = {}
        self._lock = RLock()
        for resource in resources:
            self.register(resource)

    def register(self, resource: ResourceState) -> None:
        with self._lock:
            if resource.resource_id in self._resources:
                raise ValueError(f"duplicate resource: {resource.resource_id}")
            self._resources[resource.resource_id] = resource

    def get(self, resource_id: str) -> ResourceState:
        try:
            return self._resources[str(resource_id).upper()]
        except KeyError as exc:
            raise KeyError(f"unknown resource: {resource_id}") from exc

    @property
    def states(self) -> dict[str, ResourceState]:
        return self._resources

    def candidates(
        self,
        eligible_resources: Iterable[str],
        task_type: str,
        required_tool: str | None,
    ) -> list[ResourceState]:
        result = []
        for resource_id in eligible_resources:
            resource = self.get(resource_id)
            if resource.available and resource.supports(task_type, required_tool):
                result.append(resource)
        return sorted(result, key=lambda item: (item.estimated_available_time, item.resource_id))

    def reserve(self, resource_id: str, task_id: str, now: float, zones: Iterable[str] = ()) -> bool:
        resource = self.get(resource_id)
        with self._lock:
            if resource.status is not ResourceStatus.IDLE:
                return resource.status is ResourceStatus.RESERVED and resource.current_task_id == task_id
            resource.status = ResourceStatus.RESERVED
            resource.current_task_id = str(task_id)
            resource.occupied_zones = {str(item).upper() for item in zones}
            resource.reserved_at = float(now)
            resource.last_state_change = float(now)
            return True

    def mark_busy(self, resource_id: str, task_id: str, now: float) -> None:
        resource = self.get(resource_id)
        with self._lock:
            if resource.status is not ResourceStatus.RESERVED or resource.current_task_id != task_id:
                raise RuntimeError(f"resource {resource.resource_id} is not reserved by {task_id}")
            resource.status = ResourceStatus.BUSY
            resource.busy_since = float(now)
            resource.last_state_change = float(now)

    def release(self, resource_id: str, task_id: str | None = None, now: float = 0.0) -> bool:
        resource = self.get(resource_id)
        with self._lock:
            if task_id is not None and resource.current_task_id != task_id:
                return False
            if resource.status in {ResourceStatus.FAULTED, ResourceStatus.OFFLINE}:
                resource.current_task_id = None
                resource.occupied_zones.clear()
                resource.reserved_at = None
                resource.busy_since = None
                return True
            resource.status = ResourceStatus.IDLE
            resource.current_task_id = None
            resource.occupied_zones.clear()
            resource.reserved_at = None
            resource.busy_since = None
            resource.estimated_available_time = float(now)
            resource.last_state_change = float(now)
            return True

    def fault(self, resource_id: str, fault_code: str, now: float) -> str | None:
        resource = self.get(resource_id)
        with self._lock:
            interrupted = resource.current_task_id
            resource.status = ResourceStatus.FAULTED
            resource.fault_code = str(fault_code)
            resource.last_state_change = float(now)
            return interrupted

    def begin_recovery(self, resource_id: str, now: float) -> None:
        resource = self.get(resource_id)
        with self._lock:
            if resource.status is not ResourceStatus.FAULTED:
                raise RuntimeError(f"resource {resource.resource_id} is not faulted")
            resource.status = ResourceStatus.RECOVERING
            resource.last_state_change = float(now)

    def recover(self, resource_id: str, now: float) -> None:
        resource = self.get(resource_id)
        with self._lock:
            if resource.status not in {ResourceStatus.FAULTED, ResourceStatus.RECOVERING}:
                return
            resource.status = ResourceStatus.IDLE
            resource.current_task_id = None
            resource.occupied_zones.clear()
            resource.fault_code = None
            resource.reserved_at = None
            resource.busy_since = None
            resource.estimated_available_time = float(now)
            resource.last_state_change = float(now)

    def reset(self, now: float = 0.0) -> None:
        for resource in self._resources.values():
            resource.status = ResourceStatus.IDLE
            resource.current_task_id = None
            resource.occupied_zones.clear()
            resource.fault_code = None
            resource.reserved_at = None
            resource.busy_since = None
            resource.estimated_available_time = float(now)
            resource.last_state_change = float(now)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {key: value.as_dict() for key, value in sorted(self._resources.items())}


__all__ = ["ResourceManager", "ResourceState", "ResourceStatus"]
