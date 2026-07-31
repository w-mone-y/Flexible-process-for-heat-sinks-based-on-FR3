"""Aggregate the six flexibility dimensions into one reportable view.

The competition is judged on flexibility, but until now the evidence for it was
spread across a routing compiler, a capability binder, a changeover matrix and a
fault controller — none of it visible in one place.  This module answers a single
question per dimension: **is it real, and what is the number that proves it?**

Each entry carries a ``state`` of ``FULL`` / ``PARTIAL`` / ``NONE`` and an
``evidence`` string.  ``PARTIAL`` is used honestly: process flexibility, for
example, models alternative routes and can explain why one is unavailable, but no
scheduler yet *chooses* between them.  Claiming ``FULL`` there would overstate
the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

FULL = "FULL"
PARTIAL = "PARTIAL"
NONE = "NONE"

_STATE_LABELS_ZH = {
    FULL: "已实现",
    PARTIAL: "部分实现",
    NONE: "未实现",
}


@dataclass(frozen=True, slots=True)
class FlexibilityDimension:
    """One flexibility dimension with its supporting number."""

    key: str
    label_zh: str
    state: str
    headline_zh: str
    evidence_zh: str
    metrics: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label_zh": self.label_zh,
            "state": self.state,
            "state_zh": _STATE_LABELS_ZH.get(self.state, self.state),
            "headline_zh": self.headline_zh,
            "evidence_zh": self.evidence_zh,
            "metrics": dict(self.metrics),
        }


def _product_dimension(profile_name: str) -> FlexibilityDimension:
    """Product flexibility: how many products run, and are they data-only?"""

    from .flexible import build_process_plan
    from .paths import CONFIG_DIR

    products: list[dict[str, Any]] = []
    for order_file in sorted((CONFIG_DIR / "orders").glob("order_*.yaml")):
        try:
            plan = build_process_plan(order_file)
        except Exception:  # pragma: no cover - a broken sample must not break the UI
            continue
        products.append(
            {
                "preset": plan.product.preset,
                "product_id": plan.product.product_id,
                "fin_count": len(plan.fin_targets),
                "path_count": len(plan.brazing_paths),
                "fin_pitch_mm": round(plan.product.fin_pitch_m * 1000.0, 1),
                "comb_module": plan.fixture_module.name,
                "clamping_force_n": plan.product.target_clamping_force_n,
                "source": order_file.name,
            }
        )
    executable_count = sum(item["preset"] in {"A", "B", "C"} for item in products)
    v2_limited = profile_name == "V2_DUAL_INSTALL" and executable_count < len(products)
    return FlexibilityDimension(
        key="product",
        label_zh="产品柔性",
        # Product flexibility is defined here as data-only plan generation;
        # runtime executability is reported separately and never hidden.
        state=FULL if len(products) >= 3 else PARTIAL,
        headline_zh=(
            f"{len(products)} 种 YAML 产品，V2 实体执行 {executable_count} 种"
            if v2_limited
            else f"{len(products)} 种产品全部由 YAML 驱动"
        ),
        evidence_zh=(
            "A/B/C 已接入 V2 实体流程；D 型能完成严格配置、几何和路线编译，"
            "但尚未接入 V2 实体工装，因此不把“可规划”冒充“可生产”。"
            if v2_limited
            else "新增产品 = 新增一个 product/order YAML，无需改动 Python，无需重新示教。"
        ),
        metrics={
            "product_count": len(products),
            "v2_executable_product_count": executable_count,
            "products": products,
        },
    )


def _process_and_resource_dimensions(
    profile_name: str,
) -> tuple[FlexibilityDimension, FlexibilityDimension]:
    """Process (OR routes) and resource (capability binding) flexibility."""

    from .flexible import build_preset_plan, compile_routing
    from .manufacturing_config import load_resource_config
    from .paths import CONFIG_DIR
    from .planning import (
        V1_SHALLOW_U_PROFILE,
        V2_DUAL_INSTALL_PROFILE,
        CapabilityBinder,
        default_capability_catalog,
        default_routing,
    )

    catalog = default_capability_catalog()
    routing = default_routing()
    resources, _zones = load_resource_config(CONFIG_DIR / "resources.yaml")
    profile = V2_DUAL_INSTALL_PROFILE if profile_name == "V2_DUAL_INSTALL" else V1_SHALLOW_U_PROFILE
    binder = CapabilityBinder(catalog, resources, profile=profile)
    operations = compile_routing(routing, build_preset_plan("A", quantity=1), catalog)[0]

    routes: list[dict[str, Any]] = []
    for operation in operations:
        if not operation.alternatives:
            continue
        branches = []
        for mode, result in binder.bind_alternatives(operation.alternatives).items():
            branches.append(
                {
                    "mode": mode,
                    "capability": next(
                        (item.capability for item in operation.alternatives if item.mode == mode),
                        "",
                    ),
                    "resources": list(result.resource_ids),
                    "duration_s": result.nominal_duration,
                    "available": bool(result.candidates),
                    "reasons": [reason for _rid, reason in result.rejected],
                }
            )
        routes.append(
            {
                "operation_id": operation.operation_id,
                "description_zh": operation.description_zh,
                "branches": branches,
            }
        )

    bindings: list[dict[str, Any]] = []
    multi = 0
    for operation in operations:
        result = binder.bind(
            operation.capability,
            operation.params,
            base_duration=operation.nominal_duration,
        )
        if len(result.candidates) > 1:
            multi += 1
        if not result.candidates and not result.rejected:
            continue
        bindings.append(
            {
                "operation_id": operation.operation_id,
                "capability": operation.capability,
                "candidates": [item.as_dict() for item in result.candidates],
                "rejected": [{"resource_id": rid, "reason": reason} for rid, reason in result.rejected],
            }
        )

    process = FlexibilityDimension(
        key="process",
        label_zh="工艺柔性",
        state=PARTIAL,
        headline_zh=f"{len(routes)} 道工序声明了可替代路线",
        evidence_zh=(
            "OR 分支已建模并可解释可用性（含中文拒绝原因），但调度器尚未在分支间自主择优，"
            "该能力属于后续的滚动时域优化。"
        ),
        metrics={"alternative_operation_count": len(routes), "routes": routes},
    )
    resource = FlexibilityDimension(
        key="resource",
        label_zh="资源柔性",
        state=PARTIAL,
        headline_zh=f"{multi} 道工序有多个资源候选（{profile.name} 剖面）",
        evidence_zh=(
            "能力绑定可计算多候选并给出中文拒绝原因；V2 的翅片安装已真实在线选择 Arm1/Arm3，"
            "其余工序仍由固定资源执行，因此保持“部分实现”而非理论性 FULL。"
        ),
        metrics={
            "profile": profile.name,
            "multi_candidate_operation_count": multi,
            "bindings": bindings,
        },
    )
    return process, resource


def _changeover_dimension() -> FlexibilityDimension:
    """Changeover flexibility: the competition's named metric."""

    from .changeover import (
        PLACEHOLDER_TEACHING_BASELINE,
        compare_changeover_baselines,
        required_configuration,
    )
    from .flexible import build_preset_plan
    from .planning.task_graph_builder import LEGACY_DURATIONS

    configurations = {
        preset: required_configuration(build_preset_plan(preset, quantity=1)) for preset in ("A", "B", "C")
    }
    queue = [configurations[preset] for preset in "ABABCC"]
    comparison = compare_changeover_baselines(queue, LEGACY_DURATIONS, PLACEHOLDER_TEACHING_BASELINE)
    improvements = comparison["improvements"]
    return FlexibilityDimension(
        key="changeover",
        label_zh="换型柔性",
        state=PARTIAL,
        headline_zh=("同族批量排序使换型时间减少 " f"{100 * improvements['sequencing_only_ratio']:.1f}%"),
        evidence_zh=(
            "换型动作由配置差分导出，相同工装零动作；序列相关换型时间已进入调度成本。"
            "注意：对人工示教基线的比例依赖占位数据，且需按演示时基折算；"
            "「仅排序贡献」是与时基无关的稳健口径。"
        ),
        metrics=comparison,
    )


def _disturbance_dimension(state: Mapping[str, Any]) -> FlexibilityDimension:
    """Disturbance flexibility, measured from the live snapshot."""

    from .fault_catalog import MANUAL_FAULT_CATALOG

    faults = list(state.get("faults_v2") or state.get("faults") or ())
    recoveries = list(state.get("recoveries", ()))
    recovered = sum(1 for record in faults if record.get("recovered"))
    return FlexibilityDimension(
        key="disturbance",
        label_zh="扰动柔性",
        state=FULL if state.get("ui_capabilities", {}).get("fault_injection") else PARTIAL,
        headline_zh=f"{len(MANUAL_FAULT_CATALOG)} 种故障可注入，当前恢复 {recovered}/{len(faults)}",
        evidence_zh=(
            "质量故障采用「布防—生产时物理显现—相机检出—托盘实体返回—返工—复检」闭环；"
            "设备故障采用资源隔离，并有每目标重试上限，超限转人工。"
        ),
        metrics={
            "catalog_size": len(MANUAL_FAULT_CATALOG),
            "fault_count": len(faults),
            "recovered_count": recovered,
            "recovery_rate": 0.0 if not faults else recovered / len(faults),
            "active_recoveries": [item for item in recoveries if item.get("status") == "RUNNING"],
            "physical_fault_count": len(state.get("physical_faults", ())),
        },
    )


def _volume_dimension(state: Mapping[str, Any]) -> FlexibilityDimension:
    """Volume flexibility: single pieces through full furnace batches."""

    async_line = state.get("async_line", {}) or {}
    furnace = state.get("furnace", {}) or {}
    return FlexibilityDimension(
        key="volume",
        label_zh="批量柔性",
        state=FULL,
        headline_zh=(
            f"在制 {async_line.get('active_wip', 0)}/{async_line.get('wip_limit', 0)} 托盘，"
            f"已完成 {furnace.get('completed_batches', 0)} 个炉批"
        ),
        evidence_zh="1 至 3 件订单共用同一执行主线；工艺兼容产品可跨订单拼炉。",
        metrics={
            "active_wip": async_line.get("active_wip", 0),
            "wip_limit": async_line.get("wip_limit", 0),
            "completed_batches": furnace.get("completed_batches", 0),
            "max_parallel_arms": (async_line.get("parallelism", {}) or {}).get("max_parallel_arms", 0),
        },
    )


def flexibility_report(state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the full six-dimension flexibility view.

    ``state`` is a UI/HTTP state snapshot; the static dimensions (product,
    process, resource, changeover) are derived from configuration and do not need
    a running line.
    """

    snapshot = dict(state or {})
    profile = str(snapshot.get("line_profile", "V1_STANDARD"))
    process, resource = _process_and_resource_dimensions(profile)
    dimensions = [
        _product_dimension(profile),
        process,
        resource,
        _volume_dimension(snapshot),
        _changeover_dimension(),
        _disturbance_dimension(snapshot),
    ]
    counts = {value: 0 for value in (FULL, PARTIAL, NONE)}
    for item in dimensions:
        counts[item.state] = counts.get(item.state, 0) + 1
    return {
        "line_profile": profile,
        "dimensions": [item.as_dict() for item in dimensions],
        "summary": {
            "total": len(dimensions),
            "full": counts[FULL],
            "partial": counts[PARTIAL],
            "none": counts[NONE],
        },
    }


__all__ = ["FlexibilityDimension", "flexibility_report"]
