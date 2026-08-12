"""Bounded tool-residency policy for Arm1 multi-order dispatch.

The policy only ranks *feasible* task-resource pairs.  Task dependencies,
station ownership, payload handoff and collision zones remain authoritative in
the manufacturing runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from ..planning.task_models import ManufacturingTask, TaskType

BASE_TASK_TYPES = frozenset({TaskType.PICK_BASE_PLATE, TaskType.PLACE_BASE_PLATE})
FIN_TASK_TYPES = frozenset({TaskType.PICK_FIN, TaskType.INSTALL_FIN, TaskType.REINSTALL_FIN})


@dataclass(frozen=True, slots=True)
class Arm1ToolPolicyConfig:
    max_base_microbatch: int = 2
    lookahead_seconds: float = 12.0
    starvation_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_base_microbatch < 1:
            raise ValueError("max_base_microbatch must be positive")
        if not isfinite(self.lookahead_seconds) or self.lookahead_seconds < 0:
            raise ValueError("lookahead_seconds must be finite and non-negative")
        if not isfinite(self.starvation_seconds) or self.starvation_seconds <= 0:
            raise ValueError("starvation_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class Arm1ToolSelection:
    blocked_pairs: frozenset[tuple[str, str]]
    reasons: dict[str, str]
    preferred_tool: str
    explanation_zh: str


class Arm1ToolResidencyPolicy:
    """Choose vacuum/gripper work without pre-empting a physical action."""

    def __init__(self, config: Arm1ToolPolicyConfig, *, initial_tool: str | None) -> None:
        self.config = config
        self.current_tool = initial_tool
        self.base_units_in_residency = 0
        self.active_fin_unit: str | None = None
        self.tool_switches = 0
        self.policy_blocks = 0
        self.last_explanation_zh = "等待Arm1任务"

    @staticmethod
    def _is_arm1_candidate(task: ManufacturingTask) -> bool:
        return "ARM1" in task.eligible_resources

    @staticmethod
    def _waited(task: ManufacturingTask, now: float) -> float:
        return 0.0 if task.ready_at is None else max(0.0, float(now) - task.ready_at)

    def _switch_mode(self, tool: str) -> None:
        if self.current_tool != tool:
            self.tool_switches += 1
            self.current_tool = tool

    def observe_started(self, task: ManufacturingTask, resource_id: str) -> None:
        resource = str(resource_id).upper()
        if resource == "ARM3" and task.task_type is TaskType.PICK_FIN:
            # Arm3 claimed the ready fin tray, so Arm1 may open a fresh vacuum
            # microbatch instead of waiting for work that is no longer its own.
            self.base_units_in_residency = 0
            return
        if resource != "ARM1":
            return
        if task.task_type in BASE_TASK_TYPES:
            if self.current_tool != "vacuum_gripper":
                self.base_units_in_residency = 0
            self._switch_mode("vacuum_gripper")
            return
        if task.task_type is TaskType.PREPARE_FIN_TOOL:
            if task.payload.get("arm1_tool_policy_neutral"):
                # V2 performs the visible Arm1 exchange inside the first
                # selected fin operation.  Its compatibility control node may
                # complete earlier, but must not claim gripper residency or
                # block useful vacuum work while the physical branch is still
                # undecided.
                return
            if not task.successors:
                return
            self._switch_mode("parallel_gripper")
            self.base_units_in_residency = 0
            # In V1 this node owns the following fin chain.  V2 removes those
            # dependencies for Arm3's alternative branch, so an orphan control
            # node must not reserve Arm1 for a tray Arm3 may install.
            self.active_fin_unit = task.unit_id
            return
        if task.task_type in FIN_TASK_TYPES:
            self._switch_mode("parallel_gripper")
            self.base_units_in_residency = 0
            self.active_fin_unit = task.unit_id

    def observe_succeeded(self, task: ManufacturingTask, resource_id: str, *, fin_unit_done: bool) -> None:
        if str(resource_id).upper() != "ARM1":
            return
        if task.task_type is TaskType.PLACE_BASE_PLATE:
            self.base_units_in_residency += 1
        if task.task_type in {TaskType.INSTALL_FIN, TaskType.REINSTALL_FIN} and fin_unit_done:
            if self.active_fin_unit == task.unit_id:
                self.active_fin_unit = None

    def observe_failed(self, task: ManufacturingTask, resource_id: str) -> None:
        if str(resource_id).upper() == "ARM1" and self.active_fin_unit == task.unit_id:
            self.active_fin_unit = None

    def select(
        self,
        tasks: Iterable[ManufacturingTask],
        *,
        now: float,
        resource_tool: str | None,
        next_base_ready_in: float,
        next_fin_ready_in: float,
    ) -> Arm1ToolSelection:
        values = list(tasks)
        if self.current_tool is None:
            self.current_tool = resource_tool
        arm1 = [task for task in values if self._is_arm1_candidate(task)]
        bases = [task for task in arm1 if task.task_type in BASE_TASK_TYPES]
        prepares = [task for task in arm1 if task.task_type is TaskType.PREPARE_FIN_TOOL]
        fins = [task for task in arm1 if task.task_type in FIN_TASK_TYPES]
        blocked: set[tuple[str, str]] = set()
        reasons: dict[str, str] = {}

        def block(items: Iterable[ManufacturingTask], reason: str) -> None:
            for item in items:
                blocked.add((item.task_id, "ARM1"))
                reasons[item.task_id] = reason

        if self.active_fin_unit is not None:
            block(bases, f"夹爪驻留：先完整安装{self.active_fin_unit}的全部翅片")
            block(
                (task for task in prepares if task.unit_id != self.active_fin_unit),
                f"夹爪驻留：{self.active_fin_unit}尚未完成",
            )
            block(
                (task for task in fins if task.unit_id != self.active_fin_unit),
                f"夹爪驻留：禁止跨托盘交错安装，当前为{self.active_fin_unit}",
            )
            preferred = "parallel_gripper"
            explanation = f"Arm1保持夹爪并完成{self.active_fin_unit}"
        else:
            starved_fins = [
                task for task in fins if self._waited(task, now) >= self.config.starvation_seconds
            ]
            highest_base_priority = max((task.priority for task in bases), default=-1)
            priority_fins = [task for task in fins if task.priority > highest_base_priority]
            force_gripper = bool(starved_fins or priority_fins)
            fin_window_open = bool(fins) or next_fin_ready_in <= self.config.lookahead_seconds
            at_base_limit = (
                self.base_units_in_residency >= self.config.max_base_microbatch and fin_window_open
            )

            if force_gripper or at_base_limit:
                block(bases, "吸盘微批已达上限或夹爪任务需要防饥饿")
                preferred = "parallel_gripper"
                explanation = "切换夹爪：两张基板微批完成或夹爪任务等待超限"
            elif bases:
                block((*prepares, *fins), "保持吸盘：先完成当前可执行的基板微批")
                preferred = "vacuum_gripper"
                explanation = "保持吸盘并继续当前基板微批"
            elif prepares or fins:
                imminent_base = next_base_ready_in <= self.config.lookahead_seconds
                if (
                    self.current_tool == "vacuum_gripper"
                    and imminent_base
                    and not fins
                    and (
                        self.base_units_in_residency < self.config.max_base_microbatch or not fin_window_open
                    )
                ):
                    block(prepares, f"短视窗等待：下一基板预计{next_base_ready_in:.1f}秒内可执行")
                    preferred = "vacuum_gripper"
                    explanation = "短视窗保持吸盘，避免一次无效往返换刀"
                else:
                    preferred = "parallel_gripper"
                    explanation = "吸盘暂无可执行任务，提前准备或使用夹爪"
            else:
                preferred = self.current_tool or resource_tool or "vacuum_gripper"
                explanation = "Arm1当前没有可执行任务"

        self.policy_blocks += len(blocked)
        self.last_explanation_zh = explanation
        return Arm1ToolSelection(frozenset(blocked), reasons, preferred, explanation)

    def reset(self, *, initial_tool: str | None) -> None:
        self.current_tool = initial_tool
        self.base_units_in_residency = 0
        self.active_fin_unit = None
        self.tool_switches = 0
        self.policy_blocks = 0
        self.last_explanation_zh = "等待Arm1任务"

    def snapshot(self) -> dict[str, object]:
        return {
            "current_tool": self.current_tool,
            "base_units_in_residency": self.base_units_in_residency,
            "active_fin_unit": self.active_fin_unit,
            "max_base_microbatch": self.config.max_base_microbatch,
            "lookahead_seconds": self.config.lookahead_seconds,
            "starvation_seconds": self.config.starvation_seconds,
            "tool_switches": self.tool_switches,
            "policy_blocks": self.policy_blocks,
            "explanation_zh": self.last_explanation_zh,
        }


__all__ = [
    "Arm1ToolPolicyConfig",
    "Arm1ToolResidencyPolicy",
    "Arm1ToolSelection",
    "BASE_TASK_TYPES",
    "FIN_TASK_TYPES",
]
