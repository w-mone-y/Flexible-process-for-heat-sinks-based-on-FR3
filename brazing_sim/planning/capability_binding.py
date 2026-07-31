"""Capability-based eligibility with per-line execution profiles (step B).

A task node no longer names a single resource.  It names a *capability*, and the
candidate set is derived at build/dispatch time from:

1. which resources declare that capability (``config/resources.yaml``);
2. whether a resource carries a tool of the required class;
3. whether the operation's parameters fall inside the resource's window;
4. what the *line* can physically execute.

Point 4 matters.  ``ResourceCapability`` describes the robot; a line profile
describes the scene.  In the V1 shallow-U scene the fin pick/insert skills are
implemented against ``arm1`` welds only, so even though Arm3 owns a narrow
gripper in V2, offering Arm3 as a V1 candidate would hand it a task its actor
cannot run.  Profiles keep that limitation explicit and testable instead of
letting capability data silently over-promise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..flexible.capability_models import CapabilityCatalog
from ..scheduling.resource_manager import ResourceState


@dataclass(frozen=True, slots=True)
class LineExecutionProfile:
    """Which resources a given line can actually execute a capability with."""

    name: str
    # capability -> allowed resource ids.  A capability absent from the mapping
    # is unrestricted (every declaring resource is a candidate).
    restrictions: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def allows(self, capability: str, resource_id: str) -> bool:
        allowed = self.restrictions.get(str(capability).upper())
        if allowed is None:
            return True
        return str(resource_id).upper() in allowed

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "restrictions": {key: sorted(value) for key, value in sorted(self.restrictions.items())},
        }


# The V1 scene implements fin handling against Arm1's welds and tool rack only
# (see ``async_line_skills._fin_pick_stages`` / ``_fin_place_stages``).  Arm3 is
# an inspection-only arm there.
V1_SHALLOW_U_PROFILE = LineExecutionProfile(
    name="V1_SHALLOW_U",
    restrictions={
        "FIN_PICKING": frozenset({"ARM1"}),
        "FIN_ASSEMBLY": frozenset({"ARM1"}),
        # V1 has no fixed vision gantry body; brazing inspection is Arm3's job.
        "FIXED_VISION_BRAZING": frozenset(),
    },
)

# The V2 dual-install line adds Arm3's narrow gripper, its own fin nest and a
# fixed post-braze camera, so both arms are genuine fin-assembly candidates.
V2_DUAL_INSTALL_PROFILE = LineExecutionProfile(
    name="V2_DUAL_INSTALL",
    restrictions={
        "FIXED_VISION_BRAZING": frozenset(),
    },
)

UNRESTRICTED_PROFILE = LineExecutionProfile(name="UNRESTRICTED")


@dataclass(frozen=True, slots=True)
class CapabilityCandidate:
    """One viable resource for a capability, with its speed-adjusted duration."""

    resource_id: str
    capability: str
    speed_factor: float
    duration: float
    preemptive: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "capability": self.capability,
            "speed_factor": self.speed_factor,
            "duration": self.duration,
            "preemptive": self.preemptive,
        }


@dataclass(frozen=True, slots=True)
class BindingResult:
    """Candidates plus per-resource rejection reasons for UI explainability."""

    capability: str
    candidates: tuple[CapabilityCandidate, ...]
    rejected: tuple[tuple[str, str], ...] = ()

    @property
    def resource_ids(self) -> tuple[str, ...]:
        return tuple(item.resource_id for item in self.candidates)

    @property
    def nominal_duration(self) -> float:
        """Duration of the fastest candidate, or 0 when nothing is eligible."""

        return min((item.duration for item in self.candidates), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "candidates": [item.as_dict() for item in self.candidates],
            "rejected": [{"resource_id": key, "reason": value} for key, value in self.rejected],
        }


class CapabilityBinder:
    """Resolve ``capability + params`` into a deterministic candidate list."""

    def __init__(
        self,
        catalog: CapabilityCatalog,
        resources: Iterable[ResourceState],
        *,
        profile: LineExecutionProfile = UNRESTRICTED_PROFILE,
    ) -> None:
        self.catalog = catalog
        self.profile = profile
        self.resources = {resource.resource_id: resource for resource in resources}

    def bind(
        self,
        capability: str,
        params: Mapping[str, Any] | None = None,
        *,
        base_duration: float | None = None,
    ) -> BindingResult:
        """Return every resource able to perform ``capability`` with ``params``."""

        name = str(capability).upper()
        spec = self.catalog.get(name)
        nominal = spec.duration_for(params or {}) if base_duration is None else float(base_duration)
        candidates: list[CapabilityCandidate] = []
        rejected: list[tuple[str, str]] = []
        for resource_id in sorted(self.resources):
            resource = self.resources[resource_id]
            if not resource.has_capability(name):
                continue
            if not self.profile.allows(name, resource_id):
                rejected.append((resource_id, f"当前产线（{self.profile.name}）不支持该资源执行 {name}"))
                continue
            ok, reason = resource.accepts_capability(
                name,
                params,
                tool_class=spec.requires_tool_class,
            )
            if not ok:
                rejected.append((resource_id, reason))
                continue
            declared = resource.capability(name)
            speed = 1.0 if declared is None else float(declared.speed_factor)
            if speed <= 0.0:
                rejected.append((resource_id, "节拍系数必须为正"))
                continue
            candidates.append(
                CapabilityCandidate(
                    resource_id=resource_id,
                    capability=name,
                    speed_factor=speed,
                    duration=nominal / speed,
                    # A capability is non-preemptive if either the ontology or
                    # the resource declares it so.
                    preemptive=spec.preemptive and (declared is None or declared.preemptive),
                )
            )
        # Fastest first, resource id as a deterministic tie-break.
        candidates.sort(key=lambda item: (item.duration, item.resource_id))
        return BindingResult(
            capability=name,
            candidates=tuple(candidates),
            rejected=tuple(rejected),
        )

    def bind_alternatives(
        self,
        options: Iterable[Any],
    ) -> dict[str, BindingResult]:
        """Bind every OR branch, keyed by its ``mode``.

        Branch modes with no viable resource are kept with an empty candidate
        list so the UI can explain *why* a route was unavailable rather than
        silently dropping it.
        """

        result: dict[str, BindingResult] = {}
        for option in options:
            result[str(option.mode)] = self.bind(
                option.capability,
                option.params,
                base_duration=option.nominal_duration,
            )
        return result


__all__ = [
    "BindingResult",
    "CapabilityBinder",
    "CapabilityCandidate",
    "LineExecutionProfile",
    "UNRESTRICTED_PROFILE",
    "V1_SHALLOW_U_PROFILE",
    "V2_DUAL_INSTALL_PROFILE",
]
