"""Configuration-to-execution ProcessPlan builder."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..domain import BrazingRecipe, OrderSpec
from .geometry import MAX_FINS, MAX_PATHS, generate_geometry
from .loader import (
    FlexibleConfigError,
    load_fixture_modules,
    load_order,
    load_process_recipes,
    load_product,
    load_rack_config,
)
from .models import OrderConfig, ProcessPlan, ProductConfig, RackAssignment, RackConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_ROOT = PROJECT_ROOT / "config"


def allocate_rack(
    order: OrderConfig,
    rack: RackConfig,
    *,
    occupied_layers: Iterable[int] = (),
) -> tuple[RackAssignment, ...]:
    occupied = {int(layer) for layer in occupied_layers}
    known = {layer.index for layer in rack.layers}
    if not occupied <= known:
        raise ValueError(f"占用层位{sorted(occupied - known)}不在料架配置中")
    available = [layer for layer in rack.layers if layer.index not in occupied]
    if len(available) < order.quantity:
        raise ValueError(f"料架空层不足：订单需要{order.quantity}层，仅剩{len(available)}层")
    selected = []
    preferred = order.preferred_rack_layer
    if preferred is not None:
        preferred_layer = next((layer for layer in available if layer.index == preferred), None)
        if preferred_layer is not None:
            selected.append(preferred_layer)
            available.remove(preferred_layer)
    selected.extend(sorted(available, key=lambda layer: layer.index)[: order.quantity - len(selected)])
    return tuple(
        RackAssignment(
            unit_index=index,
            tray_id=f"tray_{index + 1:02d}",
            layer_index=layer.index,
            height_m=layer.height_m,
        )
        for index, layer in enumerate(selected)
    )


def _execution_spec(product: ProductConfig, recipe: BrazingRecipe) -> OrderSpec:
    return OrderSpec(
        preset=product.preset,
        base_size=product.base_size_m,
        fin_size=product.fin_size_m,
        fin_count=product.fin_count,
        fin_pitch=product.fin_pitch_m,
        comb_module_name=product.comb_module,
        brazing_sides=product.brazing_sides,
        path_width=product.path_width_m,
        max_fins=MAX_FINS,
        max_paths=MAX_PATHS,
        recipe=recipe,
        start_offset_y=product.start_offset_y_m,
        path_margin=product.path_margin_m,
        bead_offset=product.bead_offset_m,
        nozzle_spacing=product.nozzle_spacing_m,
        nozzle_tip_height=product.nozzle_tip_height_m,
        material_speed=product.material_speed_m_s,
        target_clamping_force_n=product.target_clamping_force_n,
        clamping_force_tolerance_n=product.clamping_force_tolerance_n,
        force_hold_duration_s=product.force_hold_duration_s,
    )


def build_process_plan(
    order_file: str | Path,
    *,
    config_root: str | Path | None = None,
    occupied_layers: Iterable[int] = (),
) -> ProcessPlan:
    root = Path(config_root).expanduser().resolve() if config_root is not None else DEFAULT_CONFIG_ROOT
    order = load_order(order_file)
    product_path = (order.source_file.parent / order.product).resolve()
    product = load_product(product_path)
    modules = load_fixture_modules(root / "fixture_modules.yaml")
    recipes = load_process_recipes(root / "process_recipes.yaml")
    rack = load_rack_config(root / "rack_config.yaml")
    if product.comb_module not in modules:
        raise FlexibleConfigError(product_path, "comb_module", f"工装目录中不存在{product.comb_module}")
    module = modules[product.comb_module]
    if abs(module.pitch_m - product.fin_pitch_m) > 1.0e-9:
        raise FlexibleConfigError(product_path, "comb_module", "梳齿节距与产品翅片节距不一致")
    if module.slot_count < product.fin_count:
        raise FlexibleConfigError(product_path, "comb_module", "梳齿槽数不足")
    if product.recipe not in recipes:
        raise FlexibleConfigError(product_path, "recipe", f"工艺目录中不存在{product.recipe}")
    if abs(product.nozzle_spacing_m - 2.0 * product.bead_offset_m) > 1.0e-9:
        raise FlexibleConfigError(product_path, "nozzle_spacing_m", "必须等于2倍bead_offset_m")
    try:
        fins, paths = generate_geometry(product)
    except ValueError as exc:
        raise FlexibleConfigError(product_path, "geometry", str(exc)) from exc
    recipe = recipes[product.recipe]
    try:
        spec = _execution_spec(product, recipe.to_domain())
    except ValueError as exc:
        raise FlexibleConfigError(product_path, "process_parameters", str(exc)) from exc
    try:
        assignments = allocate_rack(order, rack, occupied_layers=occupied_layers)
    except ValueError as exc:
        raise FlexibleConfigError(order.source_file, "rack_assignment", str(exc)) from exc
    return ProcessPlan(
        order=order,
        product=product,
        execution_spec=spec,
        fin_targets=fins,
        brazing_paths=paths,
        fixture_module=module,
        recipe=recipe,
        rack_assignments=assignments,
    )


def preset_order_file(preset: str, *, config_root: str | Path | None = None) -> Path:
    root = Path(config_root).expanduser().resolve() if config_root is not None else DEFAULT_CONFIG_ROOT
    mapping = {"A": "order_001.yaml", "B": "order_002.yaml", "C": "order_003.yaml"}
    key = str(preset).upper()
    if key not in mapping:
        raise KeyError(f"unknown order preset: {preset!r}")
    return root / "orders" / mapping[key]


def build_preset_plan(
    preset: str,
    *,
    quantity: int = 1,
    config_root: str | Path | None = None,
) -> ProcessPlan:
    base = build_process_plan(preset_order_file(preset, config_root=config_root), config_root=config_root)
    if quantity == base.quantity:
        return base
    if quantity < 1 or quantity > 3:
        raise ValueError("quantity must be between 1 and 3")
    order = OrderConfig(
        schema_version=base.order.schema_version,
        order_id=f"{base.order.order_id}-Q{quantity}",
        product=base.order.product,
        quantity=quantity,
        priority=base.order.priority,
        due_time=base.order.due_time,
        preferred_rack_layer=base.order.preferred_rack_layer,
        source_file=base.order.source_file,
    )
    rack = load_rack_config(
        (Path(config_root).expanduser().resolve() if config_root is not None else DEFAULT_CONFIG_ROOT)
        / "rack_config.yaml"
    )
    return ProcessPlan(
        order=order,
        product=base.product,
        execution_spec=base.execution_spec,
        fin_targets=base.fin_targets,
        brazing_paths=base.brazing_paths,
        fixture_module=base.fixture_module,
        recipe=base.recipe,
        rack_assignments=allocate_rack(order, rack),
    )


__all__ = [
    "DEFAULT_CONFIG_ROOT",
    "allocate_rack",
    "build_preset_plan",
    "build_process_plan",
    "preset_order_file",
]
