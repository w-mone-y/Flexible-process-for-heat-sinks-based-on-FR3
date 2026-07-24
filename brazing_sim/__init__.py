"""Low-voltage cabinet heat-sink brazing simulation package."""

from .config import (
    A_ORDER_SPEC,
    B_ORDER_SPEC,
    C_ORDER_SPEC,
    DISPENSER_CONFIG,
    FIXTURE_CONFIG,
    create_batch_state,
    create_product_state,
    make_order_spec,
)
from .batch import BatchCoordinator
from .batch_transfer import BatchTransferActor
from .domain import (
    BatchStage,
    BatchState,
    OrderSpec,
    OrderStage,
    PressState,
    ProductState,
    RackShelf,
    RackShelfState,
    RackState,
    TaskSpec,
    TransferPhase,
    TransferState,
    TrayUnitPhase,
    TrayUnitState,
)
from .fixture import FixtureController, FixtureTaskActor
from .flexible import (
    BrazingPath,
    FinTarget,
    FixtureModuleConfig,
    FlexibleConfigError,
    FlexiblePreflightError,
    OrderConfig,
    ProcessPlan,
    ProcessRecipeConfig,
    ProductConfig,
    RackAssignment,
    RackConfig,
    build_preset_plan,
    build_process_plan,
    validate_process_plan,
)
from .preflight import PreflightCheckError, PreflightReport, preflight_check
from .events import EventBus, EventType, SystemEvent
from .manufacturing_runtime import ManufacturingRuntime
from .planning import ManufacturingTask
from .planning import TaskStatus as ManufacturingTaskStatus
from .planning import TaskType as ManufacturingTaskType

__all__ = [
    "A_ORDER_SPEC",
    "B_ORDER_SPEC",
    "BatchStage",
    "BatchState",
    "BatchCoordinator",
    "BatchTransferActor",
    "BrazingPath",
    "C_ORDER_SPEC",
    "DISPENSER_CONFIG",
    "FIXTURE_CONFIG",
    "FixtureController",
    "FixtureModuleConfig",
    "FixtureTaskActor",
    "FlexibleConfigError",
    "FlexiblePreflightError",
    "EventBus",
    "EventType",
    "FinTarget",
    "OrderConfig",
    "ManufacturingRuntime",
    "ManufacturingTask",
    "ManufacturingTaskStatus",
    "ManufacturingTaskType",
    "OrderSpec",
    "OrderStage",
    "PreflightCheckError",
    "PreflightReport",
    "ProcessPlan",
    "ProcessRecipeConfig",
    "ProductConfig",
    "PressState",
    "ProductState",
    "RackShelf",
    "RackShelfState",
    "RackState",
    "RackAssignment",
    "RackConfig",
    "TaskSpec",
    "SystemEvent",
    "TransferPhase",
    "TransferState",
    "TrayUnitPhase",
    "TrayUnitState",
    "__version__",
    "create_batch_state",
    "create_product_state",
    "build_preset_plan",
    "build_process_plan",
    "make_order_spec",
    "preflight_check",
    "validate_process_plan",
]

__version__ = "0.2.0"
