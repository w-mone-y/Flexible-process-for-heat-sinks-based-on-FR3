"""Fail-closed authority gate for TwinShield-RH commit windows.

The shadow scheduler may rank and explain candidates, but only this module may
turn a proposal into runtime assignments.  It deliberately consumes both the
immutable snapshot used for planning and the live READY/resource/zone view at
the commit boundary.  Any mismatch rejects the *whole* window and requests the
deterministic current scheduler fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..optimization import PlanStatus
from ..planning.task_models import ManufacturingTask, TaskStatus
from ..twin import DigitalTwinSnapshot
from .resource_manager import ResourceState, ResourceStatus
from .scheduler_base import Assignment
from .twinshield_shadow import ShadowRejection, ShadowScheduleProposal


@dataclass(frozen=True, slots=True)
class AuthorityDecision:
    """One all-or-nothing decision at a safe dispatch boundary."""

    accepted: bool
    source: str
    snapshot_fingerprint: str
    assignments: tuple[Assignment, ...] = ()
    rejections: tuple[ShadowRejection, ...] = ()
    fallback_reason: str = ""
    proposal_status: str = ""
    objective_value: float | None = None
    optimality_gap: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "source": self.source,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "assignments": [assignment.as_dict() for assignment in self.assignments],
            "rejections": [rejection.as_dict() for rejection in self.rejections],
            "fallback_reason": self.fallback_reason,
            "proposal_status": self.proposal_status,
            "objective_value": self.objective_value,
            "optimality_gap": self.optimality_gap,
            "metadata": dict(self.metadata),
        }


class TwinShieldAuthority:
    """Validate a short TwinShield proposal against the live runtime boundary."""

    def __init__(self, *, maximum_parallel_tasks: int = 3) -> None:
        if maximum_parallel_tasks <= 0:
            raise ValueError("maximum_parallel_tasks must be positive")
        self.maximum_parallel_tasks = int(maximum_parallel_tasks)

    @staticmethod
    def _reject(
        proposal: ShadowScheduleProposal,
        snapshot: DigitalTwinSnapshot,
        rejections: Iterable[ShadowRejection],
    ) -> AuthorityDecision:
        items = tuple(rejections)
        reason = items[0].reason_zh if items else "TwinShield候选未通过提交门控"
        return AuthorityDecision(
            accepted=False,
            source="CURRENT_SCHEDULER",
            snapshot_fingerprint=snapshot.fingerprint,
            rejections=items,
            fallback_reason=reason,
            proposal_status=proposal.status.value,
            objective_value=proposal.objective_value,
            optimality_gap=proposal.optimality_gap,
            metadata={"selected_count": proposal.selected_count},
        )

    def decide(
        self,
        proposal: ShadowScheduleProposal,
        *,
        snapshot: DigitalTwinSnapshot,
        ready_tasks: Iterable[ManufacturingTask],
        resources: Mapping[str, ResourceState],
        zone_leases: Mapping[str, Mapping[str, Any] | None],
    ) -> AuthorityDecision:
        """Return assignments only when every candidate can commit together."""

        if proposal.snapshot_fingerprint != snapshot.fingerprint:
            return self._reject(
                proposal,
                snapshot,
                (
                    ShadowRejection(
                        "",
                        None,
                        "STALE_SNAPSHOT",
                        "调度快照已变化，使用当前调度器重新选择",
                    ),
                ),
            )
        if proposal.status is not PlanStatus.FEASIBLE:
            return self._reject(
                proposal,
                snapshot,
                (
                    ShadowRejection(
                        "",
                        None,
                        "PROPOSAL_NOT_FEASIBLE",
                        proposal.message or "TwinShield没有形成可行计划",
                    ),
                ),
            )
        if proposal.validation is None or not proposal.validation.valid:
            return self._reject(
                proposal,
                snapshot,
                (
                    ShadowRejection(
                        "",
                        None,
                        "VALIDATION_FAILED",
                        "TwinShield计划未通过独立校验",
                    ),
                ),
            )
        if not proposal.selected:
            return self._reject(
                proposal,
                snapshot,
                (ShadowRejection("", None, "EMPTY_WINDOW", "没有可原子提交的安全任务"),),
            )
        if len(proposal.selected) > self.maximum_parallel_tasks:
            return self._reject(
                proposal,
                snapshot,
                (
                    ShadowRejection(
                        "",
                        None,
                        "WINDOW_CAPACITY",
                        f"承诺窗口超过{self.maximum_parallel_tasks}个并行任务上限",
                    ),
                ),
            )

        ready = {task.task_id: task for task in ready_tasks}
        used_tasks: set[str] = set()
        used_resources: set[str] = set()
        used_zones: set[str] = set()
        used_stations: set[str] = set()
        assignments: list[Assignment] = []
        rejected: list[ShadowRejection] = []

        for candidate in proposal.selected:
            task = ready.get(candidate.task_id)
            resource_id = candidate.resource_id.upper()
            if task is None or task.status is not TaskStatus.READY:
                rejected.append(
                    ShadowRejection(
                        candidate.task_id,
                        resource_id,
                        "TASK_NOT_READY",
                        f"任务{candidate.task_id}在提交前已不再READY",
                    )
                )
                continue
            if task.task_id in used_tasks:
                rejected.append(
                    ShadowRejection(task.task_id, resource_id, "DUPLICATE_TASK", "同一任务被重复选择")
                )
            if resource_id not in task.eligible_resources:
                rejected.append(
                    ShadowRejection(
                        task.task_id,
                        resource_id,
                        "INELIGIBLE_RESOURCE",
                        f"资源{resource_id}不能执行任务{task.task_id}",
                    )
                )
            resource = resources.get(resource_id)
            if resource is None:
                rejected.append(
                    ShadowRejection(task.task_id, resource_id, "UNKNOWN_RESOURCE", f"资源{resource_id}未注册")
                )
            elif resource.status is not ResourceStatus.IDLE or resource.current_task_id is not None:
                rejected.append(
                    ShadowRejection(task.task_id, resource_id, "RESOURCE_BUSY", f"资源{resource_id}已被占用")
                )
            elif not resource.supports(task.task_type.value, task.required_tool):
                rejected.append(
                    ShadowRejection(
                        task.task_id,
                        resource_id,
                        "RESOURCE_CAPABILITY",
                        f"资源{resource_id}的能力或工具不匹配",
                    )
                )
            if resource_id in used_resources:
                rejected.append(
                    ShadowRejection(
                        task.task_id,
                        resource_id,
                        "RESOURCE_CONFLICT",
                        f"本窗口重复占用资源{resource_id}",
                    )
                )
            zones = {str(zone).upper() for zone in task.required_zones}
            conflicting_zones = zones.intersection(used_zones)
            conflicting_zones.update(
                zone
                for zone in zones
                if zone_leases.get(zone) is not None
                and zone_leases[zone].get("task_id") != task.task_id
            )
            if conflicting_zones:
                rejected.append(
                    ShadowRejection(
                        task.task_id,
                        resource_id,
                        "ZONE_CONFLICT",
                        f"共享区域冲突：{', '.join(sorted(conflicting_zones))}",
                    )
                )
            station = str(task.station_id or "").upper()
            if station and station in used_stations:
                rejected.append(
                    ShadowRejection(
                        task.task_id,
                        resource_id,
                        "STATION_CONFLICT",
                        f"本窗口重复占用工位{station}",
                    )
                )
            if candidate.estimated_start_s > snapshot.sim_time + 1.0e-9:
                rejected.append(
                    ShadowRejection(
                        task.task_id,
                        resource_id,
                        "FUTURE_START",
                        f"任务预计到{candidate.estimated_start_s:.3f}s才可启动",
                    )
                )

            used_tasks.add(task.task_id)
            used_resources.add(resource_id)
            used_zones.update(zones)
            if station:
                used_stations.add(station)
            assignments.append(
                Assignment(
                    task.task_id,
                    resource_id,
                    candidate.total_cost,
                    dict(candidate.cost_components),
                )
            )

        if rejected:
            return self._reject(proposal, snapshot, rejected)
        return AuthorityDecision(
            accepted=True,
            source="TWINSHIELD_RH",
            snapshot_fingerprint=snapshot.fingerprint,
            assignments=tuple(assignments),
            proposal_status=proposal.status.value,
            objective_value=proposal.objective_value,
            optimality_gap=proposal.optimality_gap,
            metadata={"selected_count": proposal.selected_count},
        )


__all__ = ["AuthorityDecision", "TwinShieldAuthority"]
