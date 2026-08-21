"""Fail-closed admission for plans entering the physical V2 line."""

from __future__ import annotations

from collections.abc import Mapping

from ..flexible import FlexiblePreflightError, ProcessPlan, validate_process_plan
from ..flexible.multi_order import MAX_ORDER_ID_LENGTH
from ..manufacturing_config import load_resource_config
from ..paths import CONFIG_DIR
from ..planning import (
    TaskGraph,
    ProcessPlanTaskGraphBuilder,
    V2_DUAL_INSTALL_PROFILE,
    default_capability_catalog,
    default_routing,
)
from .process_geometry import V2ProcessGeometry


def build_v2_task_graph(plan: ProcessPlan):
    """Compile one plan with the exact V2 capability and actor profile."""

    resources, _zones = load_resource_config(CONFIG_DIR / "resources.yaml")
    return ProcessPlanTaskGraphBuilder(
        flexible_cell=True,
        camera_coordination=True,
        catalog=default_capability_catalog(),
        routing=default_routing(),
        resources=resources,
        profile=V2_DUAL_INSTALL_PROFILE,
    ).build(plan)


def validate_v2_order_id(order_id: object) -> str:
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("V2订单ID不能为空")
    identifier = order_id.strip()
    if len(identifier) > MAX_ORDER_ID_LENGTH:
        raise ValueError(f"V2订单ID长度不能超过{MAX_ORDER_ID_LENGTH}个字符")
    if any(ord(character) < 32 or ord(character) == 127 for character in identifier):
        raise ValueError("V2订单ID不能包含控制字符")
    return identifier


def validate_v2_plan(plan: ProcessPlan) -> TaskGraph:
    """Reject plans that cannot be represented by the six-tray V2 cell."""

    if not isinstance(plan, ProcessPlan):
        raise ValueError("V2订单必须携带ProcessPlan")
    validate_v2_order_id(plan.order.order_id)
    if isinstance(plan.quantity, bool) or not isinstance(plan.quantity, int) or not 1 <= plan.quantity <= 3:
        raise ValueError("V2订单数量必须是1到3的整数（物理托盘池/炉层上限）")
    if (
        isinstance(plan.order.priority, bool)
        or not isinstance(plan.order.priority, int)
        or plan.order.priority < 0
    ):
        raise ValueError("V2订单优先级必须是非负整数")

    try:
        validate_process_plan(plan)
        V2ProcessGeometry.from_plan(plan)
    except (FlexiblePreflightError, TypeError, ValueError) as exc:
        raise ValueError(f"V2自定义规格或托盘池校验失败：{exc}") from exc

    assignment_layers = [assignment.layer_index for assignment in plan.rack_assignments]
    if len(plan.rack_assignments) != plan.quantity or len(set(assignment_layers)) != len(assignment_layers):
        raise ValueError("V2订单的物理托盘/炉层分配数量不匹配或存在重复")
    if any(layer not in {0, 1, 2} for layer in assignment_layers):
        raise ValueError("V2订单只能使用三层物理炉架")

    try:
        graph = build_v2_task_graph(plan)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"V2能力/工艺路线校验失败：{exc}") from exc

    for task in graph:
        payload = task.payload
        warning = payload.get("capability_binding_warning")
        if warning:
            raise ValueError(f"V2能力未接通：{task.task_id}：{warning}")
        choices = payload.get("capability_choices")
        if isinstance(choices, list):
            if (
                not any(isinstance(choice, Mapping) and bool(choice.get("candidates")) for choice in choices)
                and not task.eligible_resources
            ):
                capability = str(payload.get("capability") or task.task_type.value)
                raise ValueError(f"V2工艺路线 {task.task_id} 的能力 {capability} 没有可执行分支")
        elif (
            payload.get("capability")
            and not payload.get("capability_candidates")
            and not task.eligible_resources
        ):
            raise ValueError(f"V2工艺路线 {task.task_id} 的能力 {payload['capability']} 没有可执行资源")
    return graph


__all__ = ["MAX_ORDER_ID_LENGTH", "build_v2_task_graph", "validate_v2_order_id", "validate_v2_plan"]
