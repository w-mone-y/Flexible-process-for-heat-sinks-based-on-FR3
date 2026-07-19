"""Strict YAML loaders with Chinese, source-aware diagnostics."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from ..domain import BrazingSide
from .models import (
    FixtureModuleConfig,
    OrderConfig,
    ProcessRecipeConfig,
    ProductConfig,
    RackConfig,
    RackLayerConfig,
)


class FlexibleConfigError(ValueError):
    """A deterministic configuration failure safe to expose through CLI/API."""

    def __init__(
        self,
        source: str | Path,
        field: str,
        reason: str,
        *,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        self.source = Path(source)
        self.field = field
        self.reason = reason
        self.line = line
        self.column = column
        location = str(self.source)
        if line is not None:
            location += f":{line}"
            if column is not None:
                location += f":{column}"
        path = field or "<root>"
        super().__init__(f"{location} [{path}]：{reason}")


def _load_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FlexibleConfigError(source, "<root>", "配置文件不存在")
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise FlexibleConfigError(source, "<root>", f"无法读取配置：{exc}") from exc
    try:
        value = yaml.safe_load(text)
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        raise FlexibleConfigError(
            source,
            "<yaml>",
            f"YAML语法错误：{exc.problem or str(exc)}",
            line=None if mark is None else mark.line + 1,
            column=None if mark is None else mark.column + 1,
        ) from exc
    if not isinstance(value, dict):
        raise FlexibleConfigError(source, "<root>", "顶层必须是映射对象")
    return source, value


def _strict_keys(
    source: Path,
    data: Mapping[str, Any],
    *,
    field: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    unknown = set(data) - required - optional
    missing = required - set(data)
    if unknown:
        name = sorted(unknown)[0]
        path = f"{field}.{name}" if field else name
        raise FlexibleConfigError(source, path, "未知字段，不允许静默忽略")
    if missing:
        name = sorted(missing)[0]
        path = f"{field}.{name}" if field else name
        raise FlexibleConfigError(source, path, "缺少必填字段")


def _value(
    source: Path,
    data: Mapping[str, Any],
    field: str,
    expected: type | tuple[type, ...],
    *,
    allow_none: bool = False,
    path_prefix: str = "",
) -> Any:
    path = f"{path_prefix}.{field}" if path_prefix else field
    value = data[field]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) and expected is not bool:
        raise FlexibleConfigError(source, path, f"类型错误，期望 {expected}，实际 bool")
    if not isinstance(value, expected):
        raise FlexibleConfigError(source, path, f"类型错误，期望 {expected}，实际 {type(value).__name__}")
    return value


def _number(
    source: Path,
    data: Mapping[str, Any],
    field: str,
    *,
    positive: bool = True,
    path_prefix: str = "",
) -> float:
    path = f"{path_prefix}.{field}" if path_prefix else field
    value = float(_value(source, data, field, (int, float), path_prefix=path_prefix))
    if not isfinite(value):
        raise FlexibleConfigError(source, path, "必须是有限数值")
    if positive and value <= 0.0:
        raise FlexibleConfigError(source, path, "必须大于0")
    return value


def _integer(
    source: Path,
    data: Mapping[str, Any],
    field: str,
    *,
    minimum: int = 0,
    path_prefix: str = "",
) -> int:
    path = f"{path_prefix}.{field}" if path_prefix else field
    value = _value(source, data, field, int, path_prefix=path_prefix)
    if value < minimum:
        raise FlexibleConfigError(source, path, f"必须不小于{minimum}")
    return int(value)


def _vec3(source: Path, data: Mapping[str, Any], field: str) -> tuple[float, float, float]:
    value = _value(source, data, field, list)
    if len(value) != 3:
        raise FlexibleConfigError(source, field, "必须包含3个数值（X/Y/Z）")
    converted: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise FlexibleConfigError(source, f"{field}[{index}]", "必须是数值")
        if float(item) <= 0.0:
            raise FlexibleConfigError(source, f"{field}[{index}]", "必须大于0")
        converted.append(float(item))
    return converted[0], converted[1], converted[2]


def load_product(path: str | Path) -> ProductConfig:
    source, data = _load_mapping(path)
    fields = {
        "schema_version",
        "product_id",
        "preset",
        "base_size_m",
        "fin_size_m",
        "fin_count",
        "fin_pitch_m",
        "start_offset_y_m",
        "path_margin_m",
        "path_width_m",
        "brazing_sides",
        "comb_module",
        "target_clamping_force_n",
        "clamping_force_tolerance_n",
        "force_hold_duration_s",
        "nozzle_spacing_m",
        "bead_offset_m",
        "nozzle_tip_height_m",
        "material_speed_m_s",
        "recipe",
    }
    _strict_keys(source, data, field="", required=fields)
    start = data["start_offset_y_m"]
    if start is not None and (isinstance(start, bool) or not isinstance(start, (int, float))):
        raise FlexibleConfigError(source, "start_offset_y_m", "必须是数值或null")
    if start is not None and not isfinite(float(start)):
        raise FlexibleConfigError(source, "start_offset_y_m", "必须是有限数值或null")
    sides = _value(source, data, "brazing_sides", list)
    try:
        brazing_sides = tuple(BrazingSide(str(side).lower()) for side in sides)
    except ValueError as exc:
        raise FlexibleConfigError(source, "brazing_sides", "仅支持left/right") from exc
    if not brazing_sides or len(set(brazing_sides)) != len(brazing_sides):
        raise FlexibleConfigError(source, "brazing_sides", "至少包含一个互不重复的侧别")
    product = ProductConfig(
        schema_version=_integer(source, data, "schema_version", minimum=1),
        product_id=_value(source, data, "product_id", str),
        preset=_value(source, data, "preset", str).upper(),
        base_size_m=_vec3(source, data, "base_size_m"),
        fin_size_m=_vec3(source, data, "fin_size_m"),
        fin_count=_integer(source, data, "fin_count", minimum=1),
        fin_pitch_m=_number(source, data, "fin_pitch_m"),
        start_offset_y_m=None if start is None else float(start),
        path_margin_m=_number(source, data, "path_margin_m"),
        path_width_m=_number(source, data, "path_width_m"),
        brazing_sides=brazing_sides,
        comb_module=_value(source, data, "comb_module", str),
        target_clamping_force_n=_number(source, data, "target_clamping_force_n"),
        clamping_force_tolerance_n=_number(source, data, "clamping_force_tolerance_n"),
        force_hold_duration_s=_number(source, data, "force_hold_duration_s"),
        nozzle_spacing_m=_number(source, data, "nozzle_spacing_m"),
        bead_offset_m=_number(source, data, "bead_offset_m"),
        nozzle_tip_height_m=_number(source, data, "nozzle_tip_height_m"),
        material_speed_m_s=_number(source, data, "material_speed_m_s"),
        recipe=_value(source, data, "recipe", str),
    )
    if product.schema_version != 1:
        raise FlexibleConfigError(source, "schema_version", "当前仅支持版本1")
    if not product.product_id.strip() or not product.preset.strip():
        raise FlexibleConfigError(source, "product_id", "产品标识和preset不能为空")
    return product


def load_order(path: str | Path) -> OrderConfig:
    source, data = _load_mapping(path)
    fields = {
        "schema_version",
        "order_id",
        "product",
        "quantity",
        "priority",
        "due_time",
        "preferred_rack_layer",
    }
    _strict_keys(source, data, field="", required=fields)
    due_raw = data["due_time"]
    due: datetime | None
    if due_raw is None:
        due = None
    elif isinstance(due_raw, datetime):
        due = due_raw
    elif isinstance(due_raw, str):
        try:
            due = datetime.fromisoformat(due_raw)
        except ValueError as exc:
            raise FlexibleConfigError(source, "due_time", "必须是ISO-8601时间或null") from exc
    else:
        raise FlexibleConfigError(source, "due_time", "必须是ISO-8601时间或null")
    preferred_raw = data["preferred_rack_layer"]
    if preferred_raw is not None and (isinstance(preferred_raw, bool) or not isinstance(preferred_raw, int)):
        raise FlexibleConfigError(source, "preferred_rack_layer", "必须是整数或null")
    result = OrderConfig(
        schema_version=_integer(source, data, "schema_version", minimum=1),
        order_id=_value(source, data, "order_id", str),
        product=_value(source, data, "product", str),
        quantity=_integer(source, data, "quantity", minimum=1),
        priority=_integer(source, data, "priority", minimum=0),
        due_time=due,
        preferred_rack_layer=preferred_raw,
        source_file=source,
    )
    if result.schema_version != 1:
        raise FlexibleConfigError(source, "schema_version", "当前仅支持版本1")
    if result.quantity > 3:
        raise FlexibleConfigError(source, "quantity", "首版仅支持1至3件，不自动拆分多炉批次")
    if result.preferred_rack_layer is not None and result.preferred_rack_layer not in {0, 1, 2}:
        raise FlexibleConfigError(source, "preferred_rack_layer", "必须为0、1、2或null")
    return result


def _load_catalog(
    path: str | Path,
    key: str,
    parse: Callable[[Path, str, Mapping[str, Any]], Any],
) -> dict[str, Any]:
    source, data = _load_mapping(path)
    _strict_keys(source, data, field="", required={"schema_version", key})
    if _integer(source, data, "schema_version", minimum=1) != 1:
        raise FlexibleConfigError(source, "schema_version", "当前仅支持版本1")
    raw = data[key]
    result: dict[str, Any] = {}
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise FlexibleConfigError(source, f"{key}[{index}]", "必须是映射对象")
            parsed = parse(source, f"{key}[{index}]", item)
            name = parsed.name
            if name in result:
                raise FlexibleConfigError(source, f"{key}[{index}].name", "名称重复")
            result[name] = parsed
    elif isinstance(raw, dict):
        for name, item in raw.items():
            if not isinstance(item, dict):
                raise FlexibleConfigError(source, f"{key}.{name}", "必须是映射对象")
            parsed = parse(source, f"{key}.{name}", {"name": name, **item})
            result[parsed.name] = parsed
    else:
        raise FlexibleConfigError(source, key, "必须是列表或映射")
    return result


def load_fixture_modules(path: str | Path) -> dict[str, FixtureModuleConfig]:
    def parse(source: Path, field: str, item: Mapping[str, Any]) -> FixtureModuleConfig:
        required = {"name", "pitch_m", "slot_count", "front_body", "rear_body", "legacy"}
        _strict_keys(source, item, field=field, required=required)
        module = FixtureModuleConfig(
            name=_value(source, item, "name", str, path_prefix=field),
            pitch_m=_number(source, item, "pitch_m", path_prefix=field),
            slot_count=_integer(source, item, "slot_count", minimum=1, path_prefix=field),
            front_body=_value(source, item, "front_body", str, path_prefix=field),
            rear_body=_value(source, item, "rear_body", str, path_prefix=field),
            legacy=_value(source, item, "legacy", bool, path_prefix=field),
        )
        return module

    return _load_catalog(path, "modules", parse)


def load_process_recipes(path: str | Path) -> dict[str, ProcessRecipeConfig]:
    def parse(source: Path, field: str, item: Mapping[str, Any]) -> ProcessRecipeConfig:
        required = {
            "name",
            "ambient_c",
            "preheat_c",
            "peak_c",
            "unload_c",
            "preheat_seconds",
            "ramp_seconds",
            "soak_seconds",
            "cooling_seconds",
            "door_seconds",
        }
        _strict_keys(source, item, field=field, required=required)
        recipe = ProcessRecipeConfig(
            name=_value(source, item, "name", str, path_prefix=field),
            ambient_c=_number(source, item, "ambient_c", positive=False, path_prefix=field),
            preheat_c=_number(source, item, "preheat_c", positive=False, path_prefix=field),
            peak_c=_number(source, item, "peak_c", positive=False, path_prefix=field),
            unload_c=_number(source, item, "unload_c", positive=False, path_prefix=field),
            preheat_seconds=_number(source, item, "preheat_seconds", path_prefix=field),
            ramp_seconds=_number(source, item, "ramp_seconds", path_prefix=field),
            soak_seconds=_number(source, item, "soak_seconds", path_prefix=field),
            cooling_seconds=_number(source, item, "cooling_seconds", path_prefix=field),
            door_seconds=_number(source, item, "door_seconds", path_prefix=field),
        )
        try:
            recipe.to_domain()
        except ValueError as exc:
            raise FlexibleConfigError(source, field, str(exc)) from exc
        return recipe

    return _load_catalog(path, "recipes", parse)


def load_rack_config(path: str | Path) -> RackConfig:
    source, data = _load_mapping(path)
    _strict_keys(source, data, field="", required={"schema_version", "policy", "layers"})
    if _integer(source, data, "schema_version", minimum=1) != 1:
        raise FlexibleConfigError(source, "schema_version", "当前仅支持版本1")
    policy = _value(source, data, "policy", str).upper()
    if policy != "LOWEST_EMPTY":
        raise FlexibleConfigError(source, "policy", "当前仅支持LOWEST_EMPTY")
    raw_layers = _value(source, data, "layers", list)
    layers: list[RackLayerConfig] = []
    for position, item in enumerate(raw_layers):
        field = f"layers[{position}]"
        if not isinstance(item, dict):
            raise FlexibleConfigError(source, field, "必须是映射对象")
        _strict_keys(source, item, field=field, required={"index", "height_m"})
        height_m = _number(source, item, "height_m", positive=False, path_prefix=field)
        if height_m < 0.0:
            raise FlexibleConfigError(source, f"{field}.height_m", "必须不小于0")
        layers.append(
            RackLayerConfig(
                index=_integer(source, item, "index", minimum=0, path_prefix=field),
                height_m=height_m,
            )
        )
    if [layer.index for layer in layers] != list(range(len(layers))):
        raise FlexibleConfigError(source, "layers", "层号必须从0开始连续排列")
    if len(layers) != 3:
        raise FlexibleConfigError(source, "layers", "当前物理料架必须配置3层")
    if any(right.height_m <= left.height_m for left, right in zip(layers, layers[1:])):
        raise FlexibleConfigError(source, "layers", "层高必须严格递增")
    return RackConfig(policy=policy, layers=tuple(layers))


__all__ = [
    "FlexibleConfigError",
    "load_fixture_modules",
    "load_order",
    "load_process_recipes",
    "load_product",
    "load_rack_config",
]
