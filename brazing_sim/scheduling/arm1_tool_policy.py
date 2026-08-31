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
from .rolling_horizon import HorizonAction, HorizonDecisionContext, RollingHorizonPlanner

BASE_TASK_TYPES = frozenset({TaskType.PICK_BASE_PLATE, TaskType.PLACE_BASE_PLATE})
FIN_TASK_TYPES = frozenset({TaskType.PICK_FIN, TaskType.INSTALL_FIN, TaskType.REINSTALL_FIN})


@dataclass(frozen=True, slots=True)
class Arm1ToolPolicyConfig:
    max_base_microbatch: int = 2
    lookahead_seconds: float = 12.0
    starvation_seconds: float = 30.0
    drain_admitted_base_wave: bool = False

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
    keep_vacuum_cost: float
    switch_gripper_cost: float
    selected_action: str


@dataclass(frozen=True, slots=True)
class Arm1OpportunityContext:
    """Line-level estimates supplied by the runtime at a safe task boundary."""

    next_base_ready_in: float
    next_fin_ready_in: float
    base_work_seconds: float
    fin_work_seconds: float
    tool_change_seconds: float
    downstream_blocking_seconds: float = 0.0
    arm3_inspection_pressure: float = 0.0
    admitted_base_units_remaining: int = 0
    parallel_fin_branches: int = 1

    def __post_init__(self) -> None:
        non_negative = (
            self.next_base_ready_in,
            self.next_fin_ready_in,
            self.base_work_seconds,
            self.fin_work_seconds,
            self.tool_change_seconds,
            self.downstream_blocking_seconds,
            self.arm3_inspection_pressure,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("Arm1 opportunity estimates must be non-negative")
        if self.admitted_base_units_remaining < 0:
            raise ValueError("remaining admitted base count must be non-negative")
        if self.parallel_fin_branches < 1:
            raise ValueError("parallel fin branch count must be positive")


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
        self.last_keep_vacuum_cost = float("inf")
        self.last_switch_gripper_cost = float("inf")
        self.last_selected_action = "IDLE"
        self.last_admitted_base_units_remaining = 0
        self.last_parallel_fin_branches = 1
        self.last_base_wave_target = 1
        self.fin_wave_released = False
        self.rolling_horizon = RollingHorizonPlanner(
            horizon_seconds=max(30.0, config.lookahead_seconds * 3.0),
            maximum_candidates=4,
        )

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
            self.fin_wave_released = True

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
        opportunity: Arm1OpportunityContext,
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
        keep_vacuum_cost = float("inf")
        switch_gripper_cost = float("inf")
        selected_action = "IDLE"

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
            switch_gripper_cost = 0.0
            selected_action = "CONTINUE_FIN_UNIT"
        else:
            starved_fins = [
                task for task in fins if self._waited(task, now) >= self.config.starvation_seconds
            ]
            urgent_fins = [task for task in fins if bool(task.payload.get("urgent_order", False))]
            force_gripper = bool(starved_fins or urgent_fins)
            fin_candidates = [*prepares, *fins]
            # PREPARE_FIN_TOOL is only an optional setup node. It must not be
            # mistaken for executable fin work, otherwise Arm1 changes tools
            # immediately after every base and destroys cross-tray batching.
            fin_window_open = bool(fins) or (opportunity.next_fin_ready_in <= self.config.lookahead_seconds)
            fin_wait_s = max((self._waited(task, now) for task in fins), default=0.0)
            base_limit_pressure = (
                self.config.lookahead_seconds
                if self.base_units_in_residency >= self.config.max_base_microbatch and fin_window_open
                else 0.0
            )
            imminent_base = opportunity.next_base_ready_in <= self.config.lookahead_seconds
            base_opportunity = bool(bases) or imminent_base
            if base_opportunity:
                keep_vacuum_cost = (
                    opportunity.next_base_ready_in
                    + opportunity.base_work_seconds
                    + fin_wait_s
                    + opportunity.downstream_blocking_seconds
                    + (opportunity.arm3_inspection_pressure if fins else 0.0)
                    + base_limit_pressure
                )
            if fin_window_open:
                switch_delay = (
                    0.0
                    if (self.current_tool or resource_tool) == "parallel_gripper"
                    else opportunity.tool_change_seconds
                )
                switch_path = switch_delay + opportunity.next_fin_ready_in + opportunity.fin_work_seconds
                future_base_delay = (
                    0.0
                    if not base_opportunity
                    else max(
                        0.0,
                        switch_path - opportunity.next_base_ready_in - opportunity.base_work_seconds,
                    )
                )
                switch_gripper_cost = switch_path + future_base_delay

            # Prime every parallel fin branch and keep one upstream buffer.
            # For V2's Arm1/Arm3 branches this gives a three-base first wave;
            # the remaining blanks stay admitted and are replenished after a
            # complete fin tray frees Arm1.  The cap remains a safety bound,
            # not a fixed batch size.
            admitted_wave_size = self.base_units_in_residency + opportunity.admitted_base_units_remaining
            base_wave_target = min(
                self.config.max_base_microbatch,
                admitted_wave_size,
                (
                    1
                    if self.fin_wave_released and bool(fins)
                    else max(3, opportunity.parallel_fin_branches + 1)
                ),
            )
            drain_base_wave = (
                self.config.drain_admitted_base_wave
                and opportunity.admitted_base_units_remaining > 0
                and base_opportunity
                and self.base_units_in_residency < base_wave_target
                # A normal fin may age while the physical S1 lane is being
                # drained, but that must not strand a blank which is already
                # present or will arrive inside the short lookahead. Only an
                # explicitly urgent order pierces this wave boundary.
                and not urgent_fins
            )
            horizon_actions: list[HorizonAction] = []
            if isfinite(keep_vacuum_cost):
                horizon_actions.append(
                    HorizonAction(
                        action_id="KEEP_VACUUM",
                        action_zh="保持吸盘并处理下一块基板",
                        duration_s=opportunity.base_work_seconds,
                        projected_completion_s=keep_vacuum_cost,
                        downstream_blocking_s=opportunity.downstream_blocking_seconds,
                    )
                )
            if isfinite(switch_gripper_cost):
                horizon_actions.append(
                    HorizonAction(
                        action_id="SWITCH_GRIPPER",
                        action_zh="切换夹爪并释放翅片安装瓶颈",
                        duration_s=opportunity.tool_change_seconds + opportunity.fin_work_seconds,
                        projected_completion_s=switch_gripper_cost,
                        changeover_s=0.0,
                    )
                )
            if drain_base_wave:
                horizon_choice = self.rolling_horizon.choose(
                    (
                        HorizonAction(
                            action_id="DRAIN_BASE_WAVE",
                            action_zh="保持吸盘并排空已接纳主板波次",
                            duration_s=opportunity.base_work_seconds,
                            projected_completion_s=keep_vacuum_cost,
                        ),
                    ),
                    HorizonDecisionContext(now=now),
                ).selected_action_id
            else:
                horizon_choice = (
                    None
                    if not horizon_actions
                    else self.rolling_horizon.choose(
                        horizon_actions,
                        HorizonDecisionContext(now=now),
                    ).selected_action_id
                )
            wave_limit_reached = (
                self.config.drain_admitted_base_wave
                and self.base_units_in_residency >= base_wave_target
                and fin_window_open
            )
            choose_gripper = force_gripper or wave_limit_reached or horizon_choice == "SWITCH_GRIPPER"
            if drain_base_wave:
                block(fin_candidates, "已接纳主板波次未完成，Arm1保持吸盘连续上料")
                preferred = "vacuum_gripper"
                selected_action = "DRAIN_BASE_WAVE"
                explanation = (
                    f"保持吸盘：连续完成当前" f"{opportunity.admitted_base_units_remaining}块已接纳主板"
                )
            elif choose_gripper and fin_window_open:
                block(bases, "机会成本判断：先切夹爪可减少翅片等待与下游阻塞")
                preferred = "parallel_gripper"
                selected_action = "SWITCH_GRIPPER"
                explanation = (
                    "切换夹爪：" f"夹爪方案成本{switch_gripper_cost:.1f}低于吸盘方案{keep_vacuum_cost:.1f}"
                )
            elif bases:
                block(fin_candidates, "机会成本判断：继续吸盘对整线延误更小")
                preferred = "vacuum_gripper"
                selected_action = "KEEP_VACUUM"
                explanation = (
                    "保持吸盘：" f"吸盘方案成本{keep_vacuum_cost:.1f}低于夹爪方案{switch_gripper_cost:.1f}"
                )
            elif prepares or fins:
                if (
                    self.current_tool == "vacuum_gripper"
                    and imminent_base
                    and not fins
                    and (
                        self.base_units_in_residency < self.config.max_base_microbatch or not fin_window_open
                    )
                ):
                    block(
                        prepares,
                        f"短视窗等待：下一基板预计{opportunity.next_base_ready_in:.1f}秒内可执行",
                    )
                    preferred = "vacuum_gripper"
                    selected_action = "KEEP_VACUUM"
                    explanation = "短视窗保持吸盘，避免一次无效往返换刀"
                else:
                    preferred = "parallel_gripper"
                    selected_action = "SWITCH_GRIPPER"
                    explanation = "吸盘暂无可执行任务，提前准备或使用夹爪"
            else:
                preferred = self.current_tool or resource_tool or "vacuum_gripper"
                selected_action = "IDLE"
                explanation = "Arm1当前没有可执行任务"

        self.policy_blocks += len(blocked)
        self.last_explanation_zh = explanation
        self.last_keep_vacuum_cost = keep_vacuum_cost
        self.last_switch_gripper_cost = switch_gripper_cost
        self.last_selected_action = selected_action
        self.last_admitted_base_units_remaining = opportunity.admitted_base_units_remaining
        self.last_parallel_fin_branches = opportunity.parallel_fin_branches
        self.last_base_wave_target = base_wave_target if self.active_fin_unit is None else 0
        return Arm1ToolSelection(
            frozenset(blocked),
            reasons,
            preferred,
            explanation,
            keep_vacuum_cost,
            switch_gripper_cost,
            selected_action,
        )

    def reset(self, *, initial_tool: str | None) -> None:
        self.current_tool = initial_tool
        self.base_units_in_residency = 0
        self.active_fin_unit = None
        self.tool_switches = 0
        self.policy_blocks = 0
        self.last_explanation_zh = "等待Arm1任务"
        self.last_keep_vacuum_cost = float("inf")
        self.last_switch_gripper_cost = float("inf")
        self.last_selected_action = "IDLE"
        self.last_admitted_base_units_remaining = 0
        self.last_parallel_fin_branches = 1
        self.last_base_wave_target = 1
        self.fin_wave_released = False
        self.rolling_horizon.reset()

    def snapshot(self) -> dict[str, object]:
        keep_cost = self.last_keep_vacuum_cost if isfinite(self.last_keep_vacuum_cost) else None
        switch_cost = self.last_switch_gripper_cost if isfinite(self.last_switch_gripper_cost) else None
        return {
            "current_tool": self.current_tool,
            "base_units_in_residency": self.base_units_in_residency,
            "active_fin_unit": self.active_fin_unit,
            "max_base_microbatch": self.config.max_base_microbatch,
            "drain_admitted_base_wave": self.config.drain_admitted_base_wave,
            "admitted_base_units_remaining": self.last_admitted_base_units_remaining,
            "parallel_fin_branches": self.last_parallel_fin_branches,
            "base_wave_target": self.last_base_wave_target,
            "fin_wave_released": self.fin_wave_released,
            "lookahead_seconds": self.config.lookahead_seconds,
            "starvation_seconds": self.config.starvation_seconds,
            "tool_switches": self.tool_switches,
            "policy_blocks": self.policy_blocks,
            "keep_vacuum_cost": keep_cost,
            "switch_gripper_cost": switch_cost,
            "selected_action": self.last_selected_action,
            "explanation_zh": self.last_explanation_zh,
            "rolling_horizon": dict(self.rolling_horizon.snapshot()),
        }


__all__ = [
    "Arm1ToolPolicyConfig",
    "Arm1OpportunityContext",
    "Arm1ToolResidencyPolicy",
    "Arm1ToolSelection",
    "BASE_TASK_TYPES",
    "FIN_TASK_TYPES",
]
