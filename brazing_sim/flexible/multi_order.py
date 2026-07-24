"""Strict multi-order and runtime order-plan helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

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

    def vector(name: str) -> tuple[float, float, float]:
        value = product[name]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"custom_product.{name}必须是三个数值")
        result = tuple(float(item) for item in value)
        if any(item <= 0 for item in result):
            raise ValueError(f"custom_product.{name}必须全部大于0")
        return result  # type: ignore[return-value]

    pitch = float(product["fin_pitch_m"])
    modules = load_fixture_modules(DEFAULT_CONFIG_ROOT / "fixture_modules.yaml")
    matching = [module for module in modules.values() if abs(module.pitch_m - pitch) <= 1e-9]
    if not matching:
        available = ", ".join(f"{1000 * item.pitch_m:g} mm" for item in modules.values())
        raise ValueError(f"无{1000 * pitch:g} mm实体梳齿模块；可用节距：{available}")
    module = matching[0]
    fin_count = int(product["fin_count"])
    if not 1 <= fin_count <= 12 or module.slot_count < fin_count:
        raise ValueError(f"翅片数量超出对象池或{module.name}槽位容量")
    recipes = load_process_recipes(DEFAULT_CONFIG_ROOT / "process_recipes.yaml")
    recipe_name = str(product["recipe"])
    if recipe_name not in recipes:
        raise ValueError(f"工艺配方不存在：{recipe_name}")
    nozzle_spacing = float(product["nozzle_spacing_m"])
    config = ProductConfig(
        schema_version=1,
        product_id=f"CUSTOM_{str(order_id)}",
        preset="CUSTOM",
        base_size_m=vector("base_size_m"),
        fin_size_m=vector("fin_size_m"),
        fin_count=fin_count,
        fin_pitch_m=pitch,
        start_offset_y_m=(
            None if product.get("start_offset_y_m") is None else float(product["start_offset_y_m"])
        ),
        path_margin_m=float(product["path_margin_m"]),
        path_width_m=float(product["path_width_m"]),
        brazing_sides=(BrazingSide.LEFT, BrazingSide.RIGHT),
        comb_module=module.name,
        target_clamping_force_n=float(product["target_clamping_force_n"]),
        clamping_force_tolerance_n=float(product.get("clamping_force_tolerance_n", 2.0)),
        force_hold_duration_s=1.5,
        nozzle_spacing_m=nozzle_spacing,
        bead_offset_m=0.5 * nozzle_spacing,
        nozzle_tip_height_m=float(product["nozzle_tip_height_m"]),
        material_speed_m_s=float(product["material_speed_m_s"]),
        recipe=recipe_name,
        material_system=str(product.get("material_system", "demo_brazing_material")),
    )
    fins, paths = generate_geometry(config)
    recipe = recipes[recipe_name]
    spec = _execution_spec(config, recipe.to_domain())
    if isinstance(due_time, str):
        due_time = datetime.fromisoformat(due_time)
    order = OrderConfig(
        schema_version=1,
        order_id=str(order_id),
        product="<runtime-custom>",
        quantity=int(quantity),
        priority=int(priority),
        due_time=due_time,
        preferred_rack_layer=preferred_rack_layer,
        source_file=Path("<runtime-custom>").resolve(),
    )
    rack = load_rack_config(DEFAULT_CONFIG_ROOT / "rack_config.yaml")
    return ProcessPlan(
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
