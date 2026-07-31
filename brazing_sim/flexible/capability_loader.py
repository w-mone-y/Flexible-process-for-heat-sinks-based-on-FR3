"""Strict loaders for the capability ontology and product routings.

Diagnostics follow the existing convention of :mod:`brazing_sim.flexible.loader`:
every failure names the file, the field path and a Chinese reason, so a bad
routing is rejected before any robot joint moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .capability_models import (
    BatchPolicy,
    CapabilityCatalog,
    CapabilitySpec,
    OperationAlternative,
    OperationSpec,
    ParamSpec,
    ParamType,
    ResourceCapability,
    RoutingSpec,
)
from .duration_model import DurationModel, DurationModelError
from .loader import FlexibleConfigError, _integer, _load_mapping, _number, _strict_keys, _value

_PARAM_FIELDS_REQUIRED = {"type"}
_PARAM_FIELDS_OPTIONAL = {"min", "max", "choices", "default", "required"}

_CAPABILITY_REQUIRED = {"task_type", "duration_model"}
_CAPABILITY_OPTIONAL = {
    "requires_tool_class",
    "param_schema",
    "preconditions",
    "effects",
    "zones",
    "preemptive",
    "description_zh",
}

_OPERATION_REQUIRED = {"id", "capability"}
_OPERATION_OPTIONAL = {
    "after",
    "after_previous",
    "params",
    "alternatives",
    "batchable",
    "station",
    "per_unit_of",
    "retry_limit",
    "description_zh",
}

_ALTERNATIVE_REQUIRED = {"mode", "capability"}
_ALTERNATIVE_OPTIONAL = {"cost_hint", "params"}


def _string_tuple(source: Path, data: Mapping[str, Any], field: str, path: str) -> tuple[str, ...]:
    raw = data.get(field, [])
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise FlexibleConfigError(source, path, "必须是字符串列表")
    result: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise FlexibleConfigError(source, f"{path}[{index}]", "必须是非空字符串")
        result.append(item.strip())
    duplicates = [item for item in result if result.count(item) > 1]
    if duplicates:
        raise FlexibleConfigError(source, path, f"存在重复项：{sorted(set(duplicates))}")
    return tuple(result)


def _param_spec(source: Path, name: str, raw: Any, path: str) -> ParamSpec:
    if not isinstance(raw, dict):
        raise FlexibleConfigError(source, path, "参数定义必须是映射对象")
    _strict_keys(
        source,
        raw,
        field=path,
        required=_PARAM_FIELDS_REQUIRED,
        optional=_PARAM_FIELDS_OPTIONAL,
    )
    try:
        param_type = ParamType(str(raw["type"]).lower())
    except ValueError as exc:
        allowed = [item.value for item in ParamType]
        raise FlexibleConfigError(source, f"{path}.type", f"必须是 {allowed} 之一") from exc

    minimum: float | None = None
    maximum: float | None = None
    for key, target in (("min", "minimum"), ("max", "maximum")):
        if key not in raw or raw[key] is None:
            continue
        if param_type in {ParamType.BOOL, ParamType.STRING}:
            raise FlexibleConfigError(source, f"{path}.{key}", f"{param_type.value} 类型不支持数值范围")
        bound = _number(source, raw, key, positive=False, path_prefix=path)
        if target == "minimum":
            minimum = bound
        else:
            maximum = bound
    if minimum is not None and maximum is not None and minimum > maximum:
        raise FlexibleConfigError(source, f"{path}.min", "最小值不能大于最大值")

    choices: tuple[str, ...] = ()
    if raw.get("choices") is not None:
        if param_type is not ParamType.STRING:
            raise FlexibleConfigError(source, f"{path}.choices", "仅 string 类型支持候选值")
        choices = _string_tuple(source, raw, "choices", f"{path}.choices")
        if not choices:
            raise FlexibleConfigError(source, f"{path}.choices", "候选值列表不能为空")

    required = True
    if "required" in raw and raw["required"] is not None:
        required = _value(source, raw, "required", bool, path_prefix=path)

    default = raw.get("default")
    spec = ParamSpec(
        name=name,
        type=param_type,
        minimum=minimum,
        maximum=maximum,
        choices=choices,
        default=default,
        required=required,
    )
    if default is not None:
        ok, reason = spec.check(default)
        if not ok:
            raise FlexibleConfigError(source, f"{path}.default", reason)
    return spec


def load_capabilities(path: str | Path) -> CapabilityCatalog:
    """Load and validate ``config/capabilities.yaml``."""

    source, data = _load_mapping(path)
    _strict_keys(source, data, field="", required={"schema_version", "capabilities"})
    if _integer(source, data, "schema_version", minimum=1) != 1:
        raise FlexibleConfigError(source, "schema_version", "当前仅支持版本1")
    raw_capabilities = data["capabilities"]
    if not isinstance(raw_capabilities, dict) or not raw_capabilities:
        raise FlexibleConfigError(source, "capabilities", "必须是非空映射")

    result: dict[str, CapabilitySpec] = {}
    for raw_name, raw in raw_capabilities.items():
        name = str(raw_name).strip().upper()
        path_prefix = f"capabilities.{raw_name}"
        if not name:
            raise FlexibleConfigError(source, path_prefix, "能力名称不能为空")
        if name in result:
            raise FlexibleConfigError(source, path_prefix, "能力名称重复")
        if not isinstance(raw, dict):
            raise FlexibleConfigError(source, path_prefix, "必须是映射对象")
        _strict_keys(
            source,
            raw,
            field=path_prefix,
            required=_CAPABILITY_REQUIRED,
            optional=_CAPABILITY_OPTIONAL,
        )

        param_schema: list[ParamSpec] = []
        raw_schema = raw.get("param_schema") or {}
        if not isinstance(raw_schema, dict):
            raise FlexibleConfigError(source, f"{path_prefix}.param_schema", "必须是映射对象")
        for param_name, param_raw in raw_schema.items():
            key = str(param_name).strip()
            if not key.isidentifier():
                raise FlexibleConfigError(
                    source,
                    f"{path_prefix}.param_schema.{param_name}",
                    "参数名必须是合法标识符，才能在节拍表达式中引用",
                )
            param_schema.append(_param_spec(source, key, param_raw, f"{path_prefix}.param_schema.{key}"))

        expression = _value(source, raw, "duration_model", str, path_prefix=path_prefix)
        allowed = frozenset(item.name for item in param_schema)
        try:
            duration_model = DurationModel(expression, allowed_names=allowed)
        except DurationModelError as exc:
            raise FlexibleConfigError(source, f"{path_prefix}.duration_model", str(exc)) from exc

        tool_class = raw.get("requires_tool_class")
        if tool_class is not None and (not isinstance(tool_class, str) or not tool_class.strip()):
            raise FlexibleConfigError(source, f"{path_prefix}.requires_tool_class", "必须是非空字符串或null")

        preemptive = True
        if raw.get("preemptive") is not None:
            preemptive = _value(source, raw, "preemptive", bool, path_prefix=path_prefix)

        result[name] = CapabilitySpec(
            name=name,
            task_type=_value(source, raw, "task_type", str, path_prefix=path_prefix).upper(),
            requires_tool_class=None if tool_class is None else str(tool_class).strip(),
            param_schema=tuple(param_schema),
            duration_model=duration_model,
            preconditions=_string_tuple(source, raw, "preconditions", f"{path_prefix}.preconditions"),
            effects=_string_tuple(source, raw, "effects", f"{path_prefix}.effects"),
            zones=tuple(item.upper() for item in _string_tuple(source, raw, "zones", f"{path_prefix}.zones")),
            preemptive=preemptive,
            description_zh=str(raw.get("description_zh", "")),
            source_file=source,
        )
    return CapabilityCatalog(capabilities=result, source_file=source)


def _alternative(source: Path, raw: Any, path: str) -> OperationAlternative:
    if not isinstance(raw, dict):
        raise FlexibleConfigError(source, path, "必须是映射对象")
    _strict_keys(
        source,
        raw,
        field=path,
        required=_ALTERNATIVE_REQUIRED,
        optional=_ALTERNATIVE_OPTIONAL,
    )
    cost_hint = 1.0
    if raw.get("cost_hint") is not None:
        cost_hint = _number(source, raw, "cost_hint", path_prefix=path)
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise FlexibleConfigError(source, f"{path}.params", "必须是映射对象")
    return OperationAlternative(
        mode=_value(source, raw, "mode", str, path_prefix=path).upper(),
        capability=_value(source, raw, "capability", str, path_prefix=path).upper(),
        cost_hint=cost_hint,
        params=dict(params),
    )


def load_routing(path: str | Path, *, catalog: CapabilityCatalog | None = None) -> RoutingSpec:
    """Load ``config/routings/<product>.yaml``, optionally cross-checking capabilities."""

    source, data = _load_mapping(path)
    _strict_keys(
        source,
        data,
        field="",
        required={"schema_version", "routing_id", "product", "operations"},
        optional={"description_zh"},
    )
    if _integer(source, data, "schema_version", minimum=1) != 1:
        raise FlexibleConfigError(source, "schema_version", "当前仅支持版本1")

    raw_operations = data["operations"]
    if not isinstance(raw_operations, list) or not raw_operations:
        raise FlexibleConfigError(source, "operations", "必须是非空列表")

    operations: list[OperationSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_operations):
        path_prefix = f"operations[{index}]"
        if not isinstance(raw, dict):
            raise FlexibleConfigError(source, path_prefix, "必须是映射对象")
        _strict_keys(
            source,
            raw,
            field=path_prefix,
            required=_OPERATION_REQUIRED,
            optional=_OPERATION_OPTIONAL,
        )
        operation_id = _value(source, raw, "id", str, path_prefix=path_prefix).strip().upper()
        if not operation_id:
            raise FlexibleConfigError(source, f"{path_prefix}.id", "工序编号不能为空")
        if operation_id in seen:
            raise FlexibleConfigError(source, f"{path_prefix}.id", f"工序编号重复：{operation_id}")
        seen.add(operation_id)

        after = tuple(item.upper() for item in _string_tuple(source, raw, "after", f"{path_prefix}.after"))
        for predecessor in after:
            if predecessor not in seen:
                raise FlexibleConfigError(
                    source,
                    f"{path_prefix}.after",
                    f"前置工序 {predecessor} 必须在本工序之前声明（避免环与前向引用）",
                )
        if operation_id in after:
            raise FlexibleConfigError(source, f"{path_prefix}.after", "工序不能依赖自身")

        # ``after_previous`` refers to the previous fin's node, so it may name
        # this operation itself and may point at a later-declared operation
        # without creating a cycle.  Only membership in the routing is checked,
        # after the whole file is parsed.
        after_previous = tuple(
            item.upper()
            for item in _string_tuple(source, raw, "after_previous", f"{path_prefix}.after_previous")
        )

        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise FlexibleConfigError(source, f"{path_prefix}.params", "必须是映射对象")

        raw_alternatives = raw.get("alternatives") or []
        if not isinstance(raw_alternatives, list):
            raise FlexibleConfigError(source, f"{path_prefix}.alternatives", "必须是列表")
        alternatives = tuple(
            _alternative(source, item, f"{path_prefix}.alternatives[{position}]")
            for position, item in enumerate(raw_alternatives)
        )
        modes = [item.mode for item in alternatives]
        if len(set(modes)) != len(modes):
            raise FlexibleConfigError(source, f"{path_prefix}.alternatives", "mode 不能重复")

        batchable: BatchPolicy | None = None
        raw_batch = raw.get("batchable")
        if raw_batch is not None:
            if not isinstance(raw_batch, dict):
                raise FlexibleConfigError(source, f"{path_prefix}.batchable", "必须是映射对象")
            _strict_keys(
                source,
                raw_batch,
                field=f"{path_prefix}.batchable",
                required={"group_key", "max_units"},
            )
            batchable = BatchPolicy(
                group_key=_value(source, raw_batch, "group_key", str, path_prefix=f"{path_prefix}.batchable"),
                max_units=_integer(
                    source,
                    raw_batch,
                    "max_units",
                    minimum=1,
                    path_prefix=f"{path_prefix}.batchable",
                ),
            )

        station = raw.get("station")
        if station is not None and (not isinstance(station, str) or not station.strip()):
            raise FlexibleConfigError(source, f"{path_prefix}.station", "必须是非空字符串或null")

        per_unit_of = raw.get("per_unit_of")
        if per_unit_of is not None and (not isinstance(per_unit_of, str) or not per_unit_of.strip()):
            raise FlexibleConfigError(source, f"{path_prefix}.per_unit_of", "必须是非空字符串或null")

        retry_limit = 0
        if raw.get("retry_limit") is not None:
            retry_limit = _integer(source, raw, "retry_limit", minimum=0, path_prefix=path_prefix)

        operations.append(
            OperationSpec(
                operation_id=operation_id,
                capability=_value(source, raw, "capability", str, path_prefix=path_prefix).upper(),
                after=after,
                after_previous=after_previous,
                params=dict(params),
                alternatives=alternatives,
                batchable=batchable,
                station=None if station is None else str(station).strip().upper(),
                per_unit_of=None if per_unit_of is None else str(per_unit_of).strip(),
                retry_limit=retry_limit,
                description_zh=str(raw.get("description_zh", "")),
            )
        )

    routing = RoutingSpec(
        schema_version=1,
        routing_id=_value(source, data, "routing_id", str).strip().upper(),
        product=_value(source, data, "product", str).strip().upper(),
        operations=tuple(operations),
        description_zh=str(data.get("description_zh", "")),
        source_file=source,
    )
    if not routing.routing_id or not routing.product:
        raise FlexibleConfigError(source, "routing_id", "routing_id 与 product 不能为空")

    declared = set(routing.operation_ids)
    for index, operation in enumerate(routing.operations):
        if not operation.after_previous:
            continue
        path_prefix = f"operations[{index}].after_previous"
        if operation.per_unit_of is None:
            raise FlexibleConfigError(
                source,
                path_prefix,
                "after_previous 只对按件展开的工序有意义，请同时声明 per_unit_of",
            )
        unknown = sorted(set(operation.after_previous) - declared)
        if unknown:
            raise FlexibleConfigError(source, path_prefix, f"引用了未声明的工序：{unknown}")
        for target in operation.after_previous:
            if routing.operation(target).per_unit_of != operation.per_unit_of:
                raise FlexibleConfigError(
                    source,
                    path_prefix,
                    f"工序 {target} 的 per_unit_of 与本工序不一致，无法按件配对",
                )

    if catalog is not None:
        validate_routing_against_catalog(routing, catalog)
    return routing


def validate_routing_against_catalog(routing: RoutingSpec, catalog: CapabilityCatalog) -> None:
    """Cross-check every operation, alternative and parameter against the catalog."""

    source = routing.source_file or Path(routing.routing_id)
    produced: set[str] = set()
    for index, operation in enumerate(routing.operations):
        path_prefix = f"operations[{index}]"
        if operation.capability not in catalog:
            raise FlexibleConfigError(
                source,
                f"{path_prefix}.capability",
                f"能力 {operation.capability} 未在能力本体中定义，可用能力为 {list(catalog.names)}",
            )
        capability = catalog.get(operation.capability)
        _, error = capability.normalize_params(operation.params, allow_placeholders=True)
        if error:
            raise FlexibleConfigError(source, f"{path_prefix}.params", error)

        for position, alternative in enumerate(operation.alternatives):
            alt_path = f"{path_prefix}.alternatives[{position}]"
            if alternative.capability not in catalog:
                raise FlexibleConfigError(
                    source,
                    f"{alt_path}.capability",
                    f"能力 {alternative.capability} 未在能力本体中定义",
                )
            alt_capability = catalog.get(alternative.capability)
            merged = {**dict(operation.params), **dict(alternative.params)}
            _, alt_error = alt_capability.normalize_params(merged, allow_placeholders=True)
            if alt_error:
                raise FlexibleConfigError(source, f"{alt_path}.params", alt_error)
            if set(alt_capability.effects) != set(capability.effects):
                raise FlexibleConfigError(
                    source,
                    f"{alt_path}.capability",
                    "OR 分支必须产生与主能力相同的工艺效果，否则不是等价替代路线："
                    f"{sorted(alt_capability.effects)} != {sorted(capability.effects)}",
                )

        missing = [item for item in capability.preconditions if item not in produced]
        if missing:
            raise FlexibleConfigError(
                source,
                f"{path_prefix}.after",
                f"能力 {capability.name} 的前置条件 {missing} 在此工序之前没有任何工序产生",
            )
        produced.update(capability.effects)


def parse_resource_capabilities(
    raw: Any,
    *,
    source: Path,
    path: str,
) -> tuple[ResourceCapability, ...]:
    """Parse the extended ``capabilities`` block of ``config/resources.yaml``.

    Both the legacy plain-string form (``[PICK_FIN, INSTALL_FIN]``) and the
    extended mapping form (with ``speed_factor`` / ``param_limits``) are
    accepted, so V1 configurations keep loading unchanged.
    """

    if not isinstance(raw, list):
        raise FlexibleConfigError(source, path, "必须是列表")
    result: list[ResourceCapability] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        item_path = f"{path}[{index}]"
        if isinstance(item, str):
            name = item.strip().upper()
            if not name:
                raise FlexibleConfigError(source, item_path, "能力名称不能为空")
            capability = ResourceCapability(name=name)
        elif isinstance(item, dict):
            _strict_keys(
                source,
                item,
                field=item_path,
                required={"name"},
                optional={"speed_factor", "preemptive", "param_limits"},
            )
            name = _value(source, item, "name", str, path_prefix=item_path).strip().upper()
            speed_factor = 1.0
            if item.get("speed_factor") is not None:
                speed_factor = _number(source, item, "speed_factor", path_prefix=item_path)
            preemptive = True
            if item.get("preemptive") is not None:
                preemptive = _value(source, item, "preemptive", bool, path_prefix=item_path)
            limits: dict[str, tuple[float, float]] = {}
            raw_limits = item.get("param_limits") or {}
            if not isinstance(raw_limits, dict):
                raise FlexibleConfigError(source, f"{item_path}.param_limits", "必须是映射对象")
            for key, bounds in raw_limits.items():
                bound_path = f"{item_path}.param_limits.{key}"
                if not isinstance(bounds, list) or len(bounds) != 2:
                    raise FlexibleConfigError(source, bound_path, "必须是 [最小值, 最大值] 两元素列表")
                converted: list[float] = []
                for position, value in enumerate(bounds):
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise FlexibleConfigError(source, f"{bound_path}[{position}]", "必须是数值")
                    converted.append(float(value))
                if converted[0] > converted[1]:
                    raise FlexibleConfigError(source, bound_path, "最小值不能大于最大值")
                limits[str(key)] = (converted[0], converted[1])
            capability = ResourceCapability(
                name=name,
                speed_factor=speed_factor,
                preemptive=preemptive,
                param_limits=limits,
            )
        else:
            raise FlexibleConfigError(source, item_path, "必须是字符串或映射对象")
        if capability.name in seen:
            raise FlexibleConfigError(source, item_path, f"能力重复声明：{capability.name}")
        seen.add(capability.name)
        result.append(capability)
    return tuple(result)


__all__ = [
    "load_capabilities",
    "load_routing",
    "parse_resource_capabilities",
    "validate_routing_against_catalog",
]
