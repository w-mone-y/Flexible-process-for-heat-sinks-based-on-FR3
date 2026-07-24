"""Order-parameter-driven flexible-manufacturing API."""

from .geometry import MAX_FINS, MAX_PATHS, fin_y_positions, generate_geometry
from .loader import (
    FlexibleConfigError,
    load_fixture_modules,
    load_order,
    load_process_recipes,
    load_product,
    load_rack_config,
)
from .models import (
    BrazingPath,
    FinTarget,
    FixtureModuleConfig,
    OrderConfig,
    ProcessPlan,
    ProcessRecipeConfig,
    ProductConfig,
    RackAssignment,
    RackConfig,
    RackLayerConfig,
    RouteStrategy,
)
from .multi_order import build_custom_plan, build_inline_plan, load_order_plans
from .planner import allocate_rack, build_preset_plan, build_process_plan, preset_order_file
from .preflight import FlexiblePreflightError, validate_process_plan

__all__ = [
    "BrazingPath",
    "FinTarget",
    "FixtureModuleConfig",
    "FlexibleConfigError",
    "FlexiblePreflightError",
    "MAX_FINS",
    "MAX_PATHS",
    "OrderConfig",
    "ProcessPlan",
    "ProcessRecipeConfig",
    "ProductConfig",
    "RackAssignment",
    "RackConfig",
    "RackLayerConfig",
    "RouteStrategy",
    "allocate_rack",
    "build_inline_plan",
    "build_custom_plan",
    "build_preset_plan",
    "build_process_plan",
    "fin_y_positions",
    "generate_geometry",
    "load_fixture_modules",
    "load_order",
    "load_order_plans",
    "load_process_recipes",
    "load_product",
    "load_rack_config",
    "preset_order_file",
    "validate_process_plan",
]
