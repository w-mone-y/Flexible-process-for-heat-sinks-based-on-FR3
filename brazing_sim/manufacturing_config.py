"""Strict configuration for scheduling resources, batches and fault scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .flexible.capability_loader import parse_resource_capabilities
from .flexible.capability_models import ResourceCapability
from .flexible.loader import FlexibleConfigError
from .recovery.fault_models import FaultType
from .scheduling.arm1_tool_policy import Arm1ToolPolicyConfig
from .scheduling.resource_manager import ResourceState
from .scheduling.scheduling_cost import SchedulingWeights


class ManufacturingConfigError(ValueError):
    pass


def _mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ManufacturingConfigError(f"{source}: 配置文件不存在")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        line = "" if mark is None else f":{mark.line + 1}:{mark.column + 1}"
        raise ManufacturingConfigError(f"{source}{line}: YAML语法错误：{exc.problem or exc}") from exc
    if not isinstance(value, dict):
        raise ManufacturingConfigError(f"{source}: 顶层必须是映射对象")
    return source, value


def _keys(
    source: Path,
    data: dict[str, Any],
    required: set[str],
    path: str = "",
    optional: set[str] | None = None,
) -> None:
    missing = required - set(data)
    unknown = set(data) - required - (optional or set())
    if missing:
        raise ManufacturingConfigError(f"{source} [{path or '<root>'}.{sorted(missing)[0]}]: 缺少必填字段")
    if unknown:
        raise ManufacturingConfigError(f"{source} [{path or '<root>'}.{sorted(unknown)[0]}]: 未知字段")


def _expect(source: Path, value: Any, expected: type, path: str) -> Any:
    if isinstance(value, bool) and expected is not bool:
        raise ManufacturingConfigError(f"{source} [{path}]: 类型错误，期望{expected.__name__}")
    if not isinstance(value, expected):
        raise ManufacturingConfigError(
            f"{source} [{path}]: 类型错误，期望{expected.__name__}，实际{type(value).__name__}"
        )
    return value


@dataclass(frozen=True, slots=True)
class BatchingConfig:
    mode: str
    max_wait_time: float
    allow_partial_batch: bool
    maximum_units: int


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    mode: str
    allow_parallel_tasks: bool
    max_assignments_per_tick: int
    weights: SchedulingWeights
    batching: BatchingConfig
    arm1_tool_policy: Arm1ToolPolicyConfig


@dataclass(frozen=True, slots=True)
class FaultTrigger:
    sim_time: float | None = None
    unit_id: str | None = None
    after_task_type: str | None = None


@dataclass(slots=True)
class FaultScenarioItem:
    fault_type: FaultType
    trigger: FaultTrigger
    payload: dict[str, Any]
    fired: bool = False


@dataclass(slots=True)
class FaultScenario:
    scenario_id: str
    random_seed: int
    faults: list[FaultScenarioItem]


def load_scheduler_config(path: str | Path) -> SchedulerConfig:
    source, data = _mapping(path)
    _keys(source, data, {"schema_version", "scheduler", "batching"})
    if _expect(source, data["schema_version"], int, "schema_version") != 1:
        raise ManufacturingConfigError(f"{source} [schema_version]: 当前仅支持版本1")
    raw = data["scheduler"]
    if not isinstance(raw, dict):
        raise ManufacturingConfigError(f"{source} [scheduler]: 必须是映射")
    _keys(
        source,
        raw,
        {"mode", "allow_parallel_tasks", "max_assignments_per_tick", "weights"},
        "scheduler",
        optional={"arm1_tool_policy"},
    )
    batching = data["batching"]
    if not isinstance(batching, dict):
        raise ManufacturingConfigError(f"{source} [batching]: 必须是映射")
    _keys(source, batching, {"mode", "max_wait_time", "allow_partial_batch", "maximum_units"}, "batching")
    weights = raw["weights"]
    if not isinstance(weights, dict):
        raise ManufacturingConfigError(f"{source} [scheduler.weights]: 必须是映射")
    _expect(source, raw["mode"], str, "scheduler.mode")
    _expect(source, raw["allow_parallel_tasks"], bool, "scheduler.allow_parallel_tasks")
    _expect(source, raw["max_assignments_per_tick"], int, "scheduler.max_assignments_per_tick")
    raw_arm1_policy = raw.get("arm1_tool_policy", {})
    if not isinstance(raw_arm1_policy, dict):
        raise ManufacturingConfigError(f"{source} [scheduler.arm1_tool_policy]: 必须是映射")
    _keys(
        source,
        raw_arm1_policy,
        set(),
        "scheduler.arm1_tool_policy",
        optional={
            "max_base_microbatch",
            "lookahead_seconds",
            "starvation_seconds",
            "drain_admitted_base_wave",
        },
    )
    if "max_base_microbatch" in raw_arm1_policy:
        _expect(
            source,
            raw_arm1_policy["max_base_microbatch"],
            int,
            "scheduler.arm1_tool_policy.max_base_microbatch",
        )
    if "drain_admitted_base_wave" in raw_arm1_policy:
        _expect(
            source,
            raw_arm1_policy["drain_admitted_base_wave"],
            bool,
            "scheduler.arm1_tool_policy.drain_admitted_base_wave",
        )
    for key in ("lookahead_seconds", "starvation_seconds"):
        if key in raw_arm1_policy and (
            isinstance(raw_arm1_policy[key], bool) or not isinstance(raw_arm1_policy[key], (int, float))
        ):
            raise ManufacturingConfigError(f"{source} [scheduler.arm1_tool_policy.{key}]: 必须是数值")
    _expect(source, batching["mode"], str, "batching.mode")
    _expect(source, batching["allow_partial_batch"], bool, "batching.allow_partial_batch")
    _expect(source, batching["maximum_units"], int, "batching.maximum_units")
    if isinstance(batching["max_wait_time"], bool) or not isinstance(batching["max_wait_time"], (int, float)):
        raise ManufacturingConfigError(f"{source} [batching.max_wait_time]: 必须是数值")
    try:
        arm1_tool_policy = Arm1ToolPolicyConfig(
            max_base_microbatch=int(raw_arm1_policy.get("max_base_microbatch", 2)),
            lookahead_seconds=float(raw_arm1_policy.get("lookahead_seconds", 12.0)),
            starvation_seconds=float(raw_arm1_policy.get("starvation_seconds", 30.0)),
            drain_admitted_base_wave=bool(raw_arm1_policy.get("drain_admitted_base_wave", False)),
        )
    except ValueError as exc:
        raise ManufacturingConfigError(f"{source} [scheduler.arm1_tool_policy]: {exc}") from exc
    result = SchedulerConfig(
        mode=str(raw["mode"]).upper(),
        allow_parallel_tasks=bool(raw["allow_parallel_tasks"]),
        max_assignments_per_tick=int(raw["max_assignments_per_tick"]),
        weights=SchedulingWeights.from_mapping(weights),
        batching=BatchingConfig(
            mode=str(batching["mode"]).upper(),
            max_wait_time=float(batching["max_wait_time"]),
            allow_partial_batch=bool(batching["allow_partial_batch"]),
            maximum_units=int(batching["maximum_units"]),
        ),
        arm1_tool_policy=arm1_tool_policy,
    )
    if result.mode not in {"FIXED_SEQUENCE", "DYNAMIC_PRIORITY"}:
        raise ManufacturingConfigError(f"{source} [scheduler.mode]: 不支持{result.mode}")
    if (
        result.max_assignments_per_tick < 1
        or not 1 <= result.batching.maximum_units <= 3
        or result.batching.max_wait_time < 0
    ):
        raise ManufacturingConfigError(f"{source}: 调度并行数和批次数量必须为正且批次不超过3")
    return result


def load_resource_config(path: str | Path) -> tuple[list[ResourceState], tuple[str, ...]]:
    source, data = _mapping(path)
    _keys(source, data, {"schema_version", "zones", "resources"})
    if (
        data["schema_version"] != 1
        or not isinstance(data["zones"], list)
        or not isinstance(data["resources"], list)
    ):
        raise ManufacturingConfigError(f"{source}: schema_version/zones/resources格式错误")
    resources: list[ResourceState] = []
    for index, raw in enumerate(data["resources"]):
        if not isinstance(raw, dict):
            raise ManufacturingConfigError(f"{source} [resources[{index}]]: 必须是映射")
        _keys(
            source,
            raw,
            {"resource_id", "resource_type", "current_tool", "available_tools", "capabilities"},
            f"resources[{index}]",
            optional={"process_capabilities", "tool_classes"},
        )
        for key in ("resource_id", "resource_type"):
            _expect(source, raw[key], str, f"resources[{index}].{key}")
        if raw["current_tool"] is not None:
            _expect(source, raw["current_tool"], str, f"resources[{index}].current_tool")
        for key in ("available_tools", "capabilities"):
            values = _expect(source, raw[key], list, f"resources[{index}].{key}")
            if any(not isinstance(item, str) for item in values):
                raise ManufacturingConfigError(f"{source} [resources[{index}].{key}]: 必须是字符串列表")

        # Capability-ontology view (step A/B).  Both fields are optional so any
        # existing resources.yaml keeps loading unchanged.
        tool_classes: dict[str, str] = {}
        raw_tool_classes = raw.get("tool_classes")
        if raw_tool_classes is not None:
            _expect(source, raw_tool_classes, dict, f"resources[{index}].tool_classes")
            for tool, tool_class in raw_tool_classes.items():
                path = f"resources[{index}].tool_classes.{tool}"
                _expect(source, tool_class, str, path)
                if str(tool) not in {str(item) for item in raw["available_tools"]}:
                    raise ManufacturingConfigError(f"{source} [{path}]: 工具 {tool} 不在 available_tools 中")
                tool_classes[str(tool)] = str(tool_class)

        process_capabilities: dict[str, ResourceCapability] = {}
        raw_process = raw.get("process_capabilities")
        if raw_process is not None:
            try:
                parsed = parse_resource_capabilities(
                    raw_process,
                    source=source,
                    path=f"resources[{index}].process_capabilities",
                )
            except FlexibleConfigError as exc:
                raise ManufacturingConfigError(str(exc)) from exc
            process_capabilities = {item.name: item for item in parsed}

        resources.append(
            ResourceState(
                resource_id=str(raw["resource_id"]),
                resource_type=str(raw["resource_type"]),
                current_tool=None if raw["current_tool"] is None else str(raw["current_tool"]),
                available_tools={str(item) for item in raw["available_tools"]},
                capabilities={str(item) for item in raw["capabilities"]},
                process_capabilities=process_capabilities,
                tool_classes=tool_classes,
            )
        )
    if any(not isinstance(zone, str) or not zone for zone in data["zones"]):
        raise ManufacturingConfigError(f"{source} [zones]: 必须是非空字符串列表")
    return resources, tuple(str(zone).upper() for zone in data["zones"])


def load_fault_scenario(path: str | Path) -> FaultScenario:
    source, data = _mapping(path)
    _keys(source, data, {"schema_version", "scenario_id", "random_seed", "faults"})
    if data["schema_version"] != 1 or not isinstance(data["faults"], list):
        raise ManufacturingConfigError(f"{source}: 故障场景格式错误")
    faults: list[FaultScenarioItem] = []
    for index, raw in enumerate(data["faults"]):
        if not isinstance(raw, dict):
            raise ManufacturingConfigError(f"{source} [faults[{index}]]: 必须是映射")
        _keys(source, raw, {"fault_type", "trigger", "payload"}, f"faults[{index}]")
        trigger = raw["trigger"]
        if not isinstance(trigger, dict) or not isinstance(raw["payload"], dict):
            raise ManufacturingConfigError(f"{source} [faults[{index}]]: trigger/payload必须是映射")
        unknown = set(trigger) - {"sim_time", "unit_id", "after_task_type"}
        if unknown:
            raise ManufacturingConfigError(f"{source} [faults[{index}].trigger]: 未知字段{sorted(unknown)}")
        faults.append(
            FaultScenarioItem(
                FaultType(str(raw["fault_type"]).upper()),
                FaultTrigger(
                    sim_time=None if trigger.get("sim_time") is None else float(trigger["sim_time"]),
                    unit_id=trigger.get("unit_id"),
                    after_task_type=trigger.get("after_task_type"),
                ),
                dict(raw["payload"]),
            )
        )
    return FaultScenario(str(data["scenario_id"]), int(data["random_seed"]), faults)


__all__ = [
    "BatchingConfig",
    "FaultScenario",
    "FaultScenarioItem",
    "FaultTrigger",
    "ManufacturingConfigError",
    "SchedulerConfig",
    "load_fault_scenario",
    "load_resource_config",
    "load_scheduler_config",
]
