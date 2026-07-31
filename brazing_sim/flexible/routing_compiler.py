"""Compile a ``RoutingSpec`` + ``ProcessPlan`` into concrete operation nodes.

This is the heart of step A.  Instead of a hand-written DAG, the process comes
from data:

1. ``$name`` placeholders in the routing are resolved from the process plan
   (path count, fin pitch, clamping force, recipe, …).
2. Parameters are validated against the capability's ``param_schema``.
3. ``per_unit_of: fin`` operations expand into one node per fin.
4. Nominal durations come from each capability's ``duration_model`` evaluated on
   the resolved parameters, replacing the constant ``DEFAULT_DURATIONS`` table.
5. OR ``alternatives`` are preserved on the node so resource binding — and the
   choice of *which* alternative to use — can be deferred to dispatch time
   (step B).

The compiler emits plain data (:class:`CompiledOperation`), not
:class:`~brazing_sim.planning.task_models.ManufacturingTask` objects, so the
planning layer keeps owning task identity, zones and line topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .capability_models import (
    CapabilityCatalog,
    CapabilitySpec,
    OperationSpec,
    RoutingSpec,
    is_placeholder,
    placeholder_name,
)
from .models import ProcessPlan


class RoutingCompileError(ValueError):
    """A routing could not be compiled against a concrete process plan."""


@dataclass(frozen=True, slots=True)
class AlternativeOption:
    """One resolved OR branch: capability, cost hint and validated parameters."""

    mode: str
    capability: str
    cost_hint: float
    params: Mapping[str, Any]
    nominal_duration: float
    requires_tool_class: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "capability": self.capability,
            "cost_hint": self.cost_hint,
            "params": dict(self.params),
            "nominal_duration": self.nominal_duration,
            "requires_tool_class": self.requires_tool_class,
        }


@dataclass(frozen=True, slots=True)
class CompiledOperation:
    """One concrete process step ready to become a task graph node."""

    node_id: str
    operation_id: str
    capability: str
    task_type: str
    params: Mapping[str, Any]
    nominal_duration: float
    predecessors: tuple[str, ...]
    alternatives: tuple[AlternativeOption, ...]
    requires_tool_class: str | None
    zones: tuple[str, ...]
    preconditions: tuple[str, ...]
    effects: tuple[str, ...]
    preemptive: bool
    retry_limit: int
    unit_index: int
    fin_index: int | None = None
    station: str | None = None
    batch_group_key: str | None = None
    batch_max_units: int | None = None
    description_zh: str = ""
    extra_payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_alternatives(self) -> bool:
        return len(self.alternatives) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operation_id": self.operation_id,
            "capability": self.capability,
            "task_type": self.task_type,
            "params": dict(self.params),
            "nominal_duration": self.nominal_duration,
            "predecessors": list(self.predecessors),
            "alternatives": [item.as_dict() for item in self.alternatives],
            "requires_tool_class": self.requires_tool_class,
            "zones": list(self.zones),
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "preemptive": self.preemptive,
            "retry_limit": self.retry_limit,
            "unit_index": self.unit_index,
            "fin_index": self.fin_index,
            "station": self.station,
            "batch_group_key": self.batch_group_key,
            "batch_max_units": self.batch_max_units,
            "description_zh": self.description_zh,
        }


def plan_parameter_bindings(plan: ProcessPlan) -> dict[str, Any]:
    """Values a routing may reference via ``$name``.

    Everything here is derived from the validated process plan, so a routing
    stays product-agnostic: the same file serves A, B, C and custom YAML orders.
    """

    product = plan.product
    return {
        "path_count": len(plan.brazing_paths),
        "fin_count": len(plan.fin_targets),
        "fin_pitch_m": float(product.fin_pitch_m),
        "comb_module": plan.fixture_module.name,
        "comb_pitch_m": float(plan.fixture_module.pitch_m),
        "comb_slot_count": int(plan.fixture_module.slot_count),
        "material_speed_m_s": float(product.material_speed_m_s),
        "bead_offset_m": float(product.bead_offset_m),
        "nozzle_spacing_m": float(product.nozzle_spacing_m),
        "nozzle_tip_height_m": float(product.nozzle_tip_height_m),
        "path_width_m": float(product.path_width_m),
        "path_margin_m": float(product.path_margin_m),
        "target_clamping_force_n": float(product.target_clamping_force_n),
        "clamping_force_tolerance_n": float(product.clamping_force_tolerance_n),
        "force_hold_duration_s": float(product.force_hold_duration_s),
        "recipe": plan.recipe.name,
        "material_system": product.material_system,
        "furnace_duration_s": float(plan.recipe.to_domain().process_seconds),
        "product_id": product.product_id,
        "preset": product.preset,
        "quantity": int(plan.quantity),
        "priority": int(plan.order.priority),
    }


def _resolve(
    params: Mapping[str, Any],
    bindings: Mapping[str, Any],
    *,
    operation_id: str,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if not is_placeholder(value):
            resolved[key] = value
            continue
        name = placeholder_name(value)
        if name not in bindings:
            available = sorted(bindings)
            raise RoutingCompileError(
                f"工序 {operation_id} 的参数 {key} 引用了未知计划变量 ${name}；" f"可用变量为 {available}"
            )
        resolved[key] = bindings[name]
    return resolved


def _validated(
    capability: CapabilitySpec,
    params: Mapping[str, Any],
    *,
    operation_id: str,
) -> dict[str, Any]:
    resolved, error = capability.normalize_params(params)
    if error:
        raise RoutingCompileError(f"工序 {operation_id}：{error}")
    return resolved


def _alternative_options(
    operation: OperationSpec,
    catalog: CapabilityCatalog,
    bindings: Mapping[str, Any],
    base_params: Mapping[str, Any],
) -> tuple[AlternativeOption, ...]:
    if not operation.alternatives:
        return ()
    options: list[AlternativeOption] = []
    for alternative in operation.alternatives:
        capability = catalog.get(alternative.capability)
        merged = {
            **dict(base_params),
            **_resolve(
                alternative.params,
                bindings,
                operation_id=f"{operation.operation_id}/{alternative.mode}",
            ),
        }
        # A branch may target a capability with a different schema, so validate
        # against that capability and keep only the parameters it declares.
        filtered = {key: value for key, value in merged.items() if key in capability.param_names}
        params = _validated(
            capability,
            filtered,
            operation_id=f"{operation.operation_id}/{alternative.mode}",
        )
        options.append(
            AlternativeOption(
                mode=alternative.mode,
                capability=capability.name,
                cost_hint=float(alternative.cost_hint),
                params=params,
                nominal_duration=capability.duration_for(params),
                requires_tool_class=capability.requires_tool_class,
            )
        )
    return tuple(options)


class RoutingCompiler:
    """Turn ``(routing, plan)`` into per-unit lists of compiled operations."""

    def __init__(self, catalog: CapabilityCatalog) -> None:
        self.catalog = catalog

    def compile_unit(
        self,
        routing: RoutingSpec,
        plan: ProcessPlan,
        unit_index: int,
    ) -> tuple[CompiledOperation, ...]:
        """Compile one product unit's operations, expanding per-fin steps."""

        bindings = plan_parameter_bindings(plan)
        fin_count = len(plan.fin_targets)
        # operation_id -> the node ids that realise it for this unit
        produced: dict[str, list[str]] = {}
        result: list[CompiledOperation] = []

        for operation in routing.operations:
            if operation.capability not in self.catalog:
                raise RoutingCompileError(
                    f"工序 {operation.operation_id} 引用了未定义能力 {operation.capability}"
                )
            capability = self.catalog.get(operation.capability)
            base_params = _resolve(operation.params, bindings, operation_id=operation.operation_id)
            filtered = {key: value for key, value in base_params.items() if key in capability.param_names}
            params = _validated(capability, filtered, operation_id=operation.operation_id)
            alternatives = _alternative_options(operation, self.catalog, bindings, base_params)

            if operation.per_unit_of == "fin":
                node_ids = self._expand_per_fin(
                    operation,
                    capability,
                    params,
                    alternatives,
                    plan,
                    unit_index,
                    produced,
                    result,
                    fin_count,
                )
            elif operation.per_unit_of is None:
                node_ids = [
                    self._emit(
                        operation,
                        capability,
                        params,
                        alternatives,
                        unit_index,
                        produced,
                        result,
                    )
                ]
            else:
                raise RoutingCompileError(
                    f"工序 {operation.operation_id} 的 per_unit_of={operation.per_unit_of!r} 暫不支持，"
                    "当前仅支持 fin 或 null"
                )
            produced[operation.operation_id] = node_ids
        return tuple(result)

    def _node_id(self, operation_id: str, unit_index: int, fin_index: int | None) -> str:
        suffix = "" if fin_index is None else f"_F{fin_index + 1:02d}"
        return f"U{unit_index + 1:02d}_{operation_id}{suffix}"

    def _predecessor_ids(
        self,
        operation: OperationSpec,
        produced: Mapping[str, list[str]],
        *,
        fin_index: int | None,
    ) -> tuple[str, ...]:
        """Resolve ``after`` to node ids.

        For a per-fin operation, a per-fin predecessor is matched fin-to-fin
        (fin *i*'s install waits only on fin *i*'s pick) while a whole-unit
        predecessor applies to every fin.  For a whole-unit operation, all node
        ids of a per-fin predecessor become predecessors, which is what
        ``INSPECT_FINS`` after every ``INSTALL_FIN`` needs.
        """

        ids: list[str] = []
        for predecessor in operation.after:
            candidates = produced.get(predecessor)
            if not candidates:
                raise RoutingCompileError(
                    f"工序 {operation.operation_id} 的前置工序 {predecessor} 没有生成任何节点"
                )
            if fin_index is not None and len(candidates) > 1:
                matched = [item for item in candidates if item.endswith(f"_F{fin_index + 1:02d}")]
                ids.extend(matched or candidates)
            else:
                ids.extend(candidates)
        return tuple(dict.fromkeys(ids))

    def _emit(
        self,
        operation: OperationSpec,
        capability: CapabilitySpec,
        params: Mapping[str, Any],
        alternatives: tuple[AlternativeOption, ...],
        unit_index: int,
        produced: Mapping[str, list[str]],
        result: list[CompiledOperation],
        *,
        fin_index: int | None = None,
        extra_predecessors: tuple[str, ...] = (),
        extra_payload: Mapping[str, Any] | None = None,
    ) -> str:
        node_id = self._node_id(operation.operation_id, unit_index, fin_index)
        predecessors = tuple(
            dict.fromkeys(
                (
                    *self._predecessor_ids(operation, produced, fin_index=fin_index),
                    *extra_predecessors,
                )
            )
        )
        result.append(
            CompiledOperation(
                node_id=node_id,
                operation_id=operation.operation_id,
                capability=capability.name,
                task_type=capability.task_type,
                params=dict(params),
                nominal_duration=capability.duration_for(params),
                predecessors=predecessors,
                alternatives=alternatives,
                requires_tool_class=capability.requires_tool_class,
                zones=capability.zones,
                preconditions=capability.preconditions,
                effects=capability.effects,
                preemptive=capability.preemptive,
                retry_limit=operation.retry_limit,
                unit_index=unit_index,
                fin_index=fin_index,
                station=operation.station,
                batch_group_key=None if operation.batchable is None else operation.batchable.group_key,
                batch_max_units=None if operation.batchable is None else operation.batchable.max_units,
                description_zh=operation.description_zh or capability.description_zh,
                extra_payload=dict(extra_payload or {}),
            )
        )
        return node_id

    def _expand_per_fin(
        self,
        operation: OperationSpec,
        capability: CapabilitySpec,
        params: Mapping[str, Any],
        alternatives: tuple[AlternativeOption, ...],
        plan: ProcessPlan,
        unit_index: int,
        produced: Mapping[str, list[str]],
        result: list[CompiledOperation],
        fin_count: int,
    ) -> list[str]:
        node_ids: list[str] = []
        for fin_index in range(fin_count):
            target = plan.fin_targets[fin_index]
            # ``after_previous`` links fin i to fin i-1 of the named operations.
            # This is how the routing states "one gripper, one fin at a time"
            # without the compiler guessing which steps must serialise.
            extra: list[str] = []
            if fin_index > 0:
                for target_operation in operation.after_previous:
                    extra.append(self._node_id(target_operation, unit_index, fin_index - 1))
            node_id = self._emit(
                operation,
                capability,
                params,
                alternatives,
                unit_index,
                produced,
                result,
                fin_index=fin_index,
                extra_predecessors=tuple(dict.fromkeys(extra)),
                extra_payload={
                    "fin_id": target.fin_id,
                    "target_position": target.position,
                },
            )
            node_ids.append(node_id)
        return node_ids


def compile_routing(
    routing: RoutingSpec,
    plan: ProcessPlan,
    catalog: CapabilityCatalog,
) -> dict[int, tuple[CompiledOperation, ...]]:
    """Compile every unit of a plan; keys are unit indices."""

    compiler = RoutingCompiler(catalog)
    return {
        assignment.unit_index: compiler.compile_unit(routing, plan, assignment.unit_index)
        for assignment in plan.rack_assignments
    }


__all__ = [
    "AlternativeOption",
    "CompiledOperation",
    "RoutingCompileError",
    "RoutingCompiler",
    "compile_routing",
    "plan_parameter_bindings",
]
