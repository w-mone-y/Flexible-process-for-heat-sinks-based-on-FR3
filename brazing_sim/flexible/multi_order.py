"""Strict multi-order and runtime order-plan helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..domain import BrazingSide
from .geometry import generate_geometry
from .loader import (
    FlexibleConfigError,
    load_fixture_modules,
    load_process_recipes,
    load_rack_config,
)
from .models import OrderConfig, ProcessPlan, ProductConfig, RouteStrategy
from .planner import (
    DEFAULT_CONFIG_ROOT,
    _execution_spec,
    allocate_rack,
    build_preset_plan,
    build_process_plan,
)


def _replace_order(plan: ProcessPlan, order: OrderConfig) -> ProcessPlan:
    rack = load_rack_config(DEFAULT_CONFIG_ROOT / "rack_config.yaml")
    return replace(plan, order=order, rack_assignments=allocate_rack(order, rack))


def build_inline_plan(
    *,
    preset: str,
    order_id: str,
    quantity: int,
    priority: int,
    due_time: datetime | str | None = None,
    preferred_rack_layer: int | None = None,
    route_strategy: RouteStrategy | str = RouteStrategy.STANDARD,
) -> ProcessPlan:
    plan = build_preset_plan(preset, quantity=quantity)
    if isinstance(due_time, str):
        due_time = datetime.fromisoformat(due_time)
    order = replace(
        plan.order,
        order_id=str(order_id),
        quantity=int(quantity),
        priority=int(priority),
        due_time=due_time,
        preferred_rack_layer=preferred_rack_layer,
    )
    return replace(_replace_order(plan, order), route_strategy=RouteStrategy(route_strategy))


def build_custom_plan(
    *,
    order_id: str,
    quantity: int,
    priority: int,
    product: dict[str, Any],
    due_time: datetime | str | None = None,
    preferred_rack_layer: int | None = None,
    route_strategy: RouteStrategy | str = RouteStrategy.STANDARD,
) -> ProcessPlan:
    """Build a strict runtime-only product without mutating product YAML."""

    if not isinstance(product, Mapping):
        raise ValueError("自定义产品配置必须是对象")
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("自定义订单ID不能为空")
    identifier = order_id.strip()
    if isinstance(quantity, bool) or not isinstance(quantity, int) or not 1 <= quantity <= 3:
        raise ValueError("自定义订单数量必须是1到3的整数（受V2托盘池约束）")
    if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
        raise ValueError("订单优先级必须是非负整数")
    if due_time is not None and not isinstance(due_time, (datetime, str)):
        raise ValueError("订单交期必须是ISO-8601字符串、datetime或空值")
    if (
        isinstance(preferred_rack_layer, bool)
        or preferred_rack_layer is not None
        and (
            not isinstance(preferred_rack_layer, int)
            or preferred_rack_layer not in {0, 1, 2}
        )
    ):
        raise ValueError("首选料架层必须是0、1、2或空值")

    required = {
        "base_size_m",
        "fin_size_m",
        "fin_count",
        "fin_pitch_m",
        "path_margin_m",
        "path_width_m",
        "nozzle_spacing_m",
        "nozzle_tip_height_m",
        "material_speed_m_s",
        "target_clamping_force_n",
        "recipe",
    }
    unknown = set(product).difference(
        required | {"start_offset_y_m", "material_system", "clamping_force_tolerance_n"}
    )
    missing = required.difference(product)
    if missing or unknown:
        message = f"缺少字段{sorted(missing)}" if missing else f"未知字段{sorted(unknown)}"
        raise ValueError(f"自定义产品配置错误：{message}")

    def number(name: str, *, positive: bool = True) -> float:
        value = product[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"custom_product.{name}必须是数值")
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"custom_product.{name}必须是有限数值")
        if positive and result <= 0.0:
            raise ValueError(f"custom_product.{name}必须大于0")
        return result

    def vector(name: str) -> tuple[float, float, float]:
        value = product[name]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"custom_product.{name}必须是三个数值")
        result = tuple(number_value for number_value in (
            float(item)
            if not isinstance(item, bool) and isinstance(item, (int, float))
            else float("nan")
            for item in value
        ))
        if any(not isfinite(item) for item in result):
            raise ValueError(f"custom_product.{name}必须全部是有限数值")
        if any(item <= 0.0 for item in result):
            raise ValueError(f"custom_product.{name}必须全部大于0")
        return result  # type: ignore[return-value]

    pitch = number("fin_pitch_m")
    modules = load_fixture_modules(DEFAULT_CONFIG_ROOT / "fixture_modules.yaml")
    matching = [module for module in modules.values() if abs(module.pitch_m - pitch) <= 1e-9]
    if not matching:
        available = ", ".join(f"{1000 * item.pitch_m:g} mm" for item in modules.values())
        raise ValueError(f"无{1000 * pitch:g} mm实体梳齿模块；可用节距：{available}")
    module = matching[0]
    raw_fin_count = product["fin_count"]
    if isinstance(raw_fin_count, bool) or not isinstance(raw_fin_count, int):
        raise ValueError("custom_product.fin_count必须是整数")
    fin_count = int(raw_fin_count)
    if not 1 <= fin_count <= 12 or module.slot_count < fin_count:
        raise ValueError(f"翅片数量超出对象池或{module.name}槽位容量")
    recipes = load_process_recipes(DEFAULT_CONFIG_ROOT / "process_recipes.yaml")
    recipe_value = product["recipe"]
    if not isinstance(recipe_value, str) or not recipe_value.strip():
        raise ValueError("custom_product.recipe必须是非空字符串")
    recipe_name = recipe_value.strip()
    if recipe_name not in recipes:
        raise ValueError(f"工艺配方不存在：{recipe_name}")
    nozzle_spacing = number("nozzle_spacing_m")
    start_offset = product.get("start_offset_y_m")
    if start_offset is not None:
        if isinstance(start_offset, bool) or not isinstance(start_offset, (int, float)):
            raise ValueError("custom_product.start_offset_y_m必须是数值或空值")
        start_offset = float(start_offset)
        if not isfinite(start_offset):
            raise ValueError("custom_product.start_offset_y_m必须是有限数值或空值")
    material_system = product.get("material_system", "demo_aluminum_brazing")
    if not isinstance(material_system, str) or not material_system.strip():
        raise ValueError("custom_product.material_system必须是非空字符串")
    config = ProductConfig(
        schema_version=1,
        product_id=f"CUSTOM_{identifier}",
        preset="CUSTOM",
        base_size_m=vector("base_size_m"),
        fin_size_m=vector("fin_size_m"),
        fin_count=fin_count,
        fin_pitch_m=pitch,
        start_offset_y_m=start_offset,
        path_margin_m=number("path_margin_m"),
        path_width_m=number("path_width_m"),
        brazing_sides=(BrazingSide.LEFT, BrazingSide.RIGHT),
        comb_module=module.name,
        target_clamping_force_n=number("target_clamping_force_n"),
        clamping_force_tolerance_n=(
            2.0
            if product.get("clamping_force_tolerance_n") is None
            else number("clamping_force_tolerance_n")
        ),
        force_hold_duration_s=1.5,
        nozzle_spacing_m=nozzle_spacing,
        bead_offset_m=0.5 * nozzle_spacing,
        nozzle_tip_height_m=number("nozzle_tip_height_m"),
        material_speed_m_s=number("material_speed_m_s"),
        recipe=recipe_name,
        # Runtime-custom products use the line's current aluminium material
        # system unless the caller explicitly selects another one.  This lets
        # a compatible custom order share a V2 furnace batch with A/B/C while
        # still preserving strict material-system separation when overridden.
        material_system=material_system.strip(),
    )
    fins, paths = generate_geometry(config)
    recipe = recipes[recipe_name]
    spec = _execution_spec(config, recipe.to_domain())
    if isinstance(due_time, str):
        due_time = datetime.fromisoformat(due_time)
    order = OrderConfig(
        schema_version=1,
        order_id=identifier,
        product="<runtime-custom>",
        quantity=int(quantity),
        priority=int(priority),
        due_time=due_time,
        preferred_rack_layer=preferred_rack_layer,
        source_file=Path("<runtime-custom>").resolve(),
    )
    rack = load_rack_config(DEFAULT_CONFIG_ROOT / "rack_config.yaml")
    plan = ProcessPlan(
        order=order,
        product=config,
        execution_spec=spec,
        fin_targets=fins,
        brazing_paths=paths,
        fixture_module=module,
        recipe=recipe,
        rack_assignments=allocate_rack(order, rack),
        route_strategy=RouteStrategy(route_strategy),
    )
    # Compile the real routing before any V1/V2 caller can enqueue the plan.
    # This validates capability parameter schemas (speed, force, path count,
    # etc.) at the trust boundary instead of after a physical runtime mutates.
    try:
        from ..planning import default_capability_catalog, default_routing
        from .routing_compiler import compile_routing

        compile_routing(default_routing(), plan, default_capability_catalog())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"自定义产品能力/工艺路线校验失败：{exc}") from exc
    return plan


def load_order_plans(path: str | Path) -> tuple[ProcessPlan, ...]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FlexibleConfigError(source, "<root>", "配置文件不存在")
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        raise FlexibleConfigError(
            source,
            "<yaml>",
            f"YAML语法错误：{exc.problem or exc}",
            line=None if mark is None else mark.line + 1,
            column=None if mark is None else mark.column + 1,
        ) from exc
    if not isinstance(data, dict) or set(data) != {"schema_version", "orders"}:
        raise FlexibleConfigError(source, "<root>", "仅允许schema_version和orders字段")
    if data["schema_version"] != 1 or not isinstance(data["orders"], list) or not data["orders"]:
        raise FlexibleConfigError(source, "orders", "版本必须为1且orders必须是非空列表")
    plans: list[ProcessPlan] = []
    known_ids: set[str] = set()
    allowed = {"order_file", "quantity", "priority", "due_time", "preferred_rack_layer", "urgent"}
    for index, raw in enumerate(data["orders"]):
        field = f"orders[{index}]"
        if not isinstance(raw, dict) or "order_file" not in raw:
            raise FlexibleConfigError(source, field, "必须是含order_file的映射")
        unknown = set(raw) - allowed
        if unknown:
            raise FlexibleConfigError(source, f"{field}.{sorted(unknown)[0]}", "未知字段")
        plan = build_process_plan((source.parent / str(raw["order_file"])).resolve())
        due_raw: Any = raw.get("due_time", plan.order.due_time)
        if isinstance(due_raw, str):
            try:
                due_raw = datetime.fromisoformat(due_raw)
            except ValueError as exc:
                raise FlexibleConfigError(source, f"{field}.due_time", "必须是ISO-8601时间") from exc
        quantity = int(raw.get("quantity", plan.quantity))
        priority = int(raw.get("priority", plan.order.priority))
        preferred = raw.get("preferred_rack_layer", plan.order.preferred_rack_layer)
        if not 1 <= quantity <= 3 or priority < 0 or preferred not in {None, 0, 1, 2}:
            raise FlexibleConfigError(source, field, "quantity/priority/preferred_rack_layer超出允许范围")
        order = replace(
            plan.order,
            quantity=quantity,
            priority=priority,
            due_time=due_raw,
            preferred_rack_layer=preferred,
        )
        if order.order_id in known_ids:
            raise FlexibleConfigError(source, f"{field}.order_file", "order_id重复")
        known_ids.add(order.order_id)
        plans.append(_replace_order(plan, order))
    return tuple(plans)


__all__ = ["build_custom_plan", "build_inline_plan", "load_order_plans"]
