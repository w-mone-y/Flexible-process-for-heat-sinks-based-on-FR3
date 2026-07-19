"""Pre-execution checks that remain usable without importing MuJoCo."""

from __future__ import annotations

from typing import Any

from .models import ProcessPlan


class FlexiblePreflightError(RuntimeError):
    pass


def validate_process_plan(plan: ProcessPlan, scene: Any | None = None) -> None:
    errors: list[str] = []
    if len(plan.fin_targets) > plan.max_fins:
        errors.append(f"翅片数{len(plan.fin_targets)}超过对象池{plan.max_fins}")
    if len(plan.brazing_paths) > plan.max_paths:
        errors.append(f"路径数{len(plan.brazing_paths)}超过对象池{plan.max_paths}")
    if plan.fixture_module.slot_count < len(plan.fin_targets):
        errors.append("选中梳齿模块的槽位不足")
    assigned = [item.layer_index for item in plan.rack_assignments]
    if len(assigned) != plan.quantity or len(set(assigned)) != len(assigned):
        errors.append("料架分配数量错误或层位重复")
    if any(layer not in {0, 1, 2} for layer in assigned):
        errors.append("料架层位超出三层物理容量")
    if scene is not None:
        model = scene.model
        required_bodies = [
            *(f"fin_{index:02d}" for index in range(1, 13)),
            *(
                f"slot_{index:02d}_{side}_brazing_path"
                for index in range(1, 13)
                for side in ("left", "right")
            ),
            plan.fixture_module.front_body,
            plan.fixture_module.rear_body,
            *(f"batch_tray_{index:02d}" for index in range(1, 4)),
        ]
        required_sites = [
            *(f"fin_slot_{index:02d}_target" for index in range(1, 13)),
            *(f"raw_fin_{index:02d}_site" for index in range(1, 13)),
            "arm2_dispenser_center_tcp",
            "arm2_left_nozzle_tip_site",
            "arm2_right_nozzle_tip_site",
        ]
        required_welds = [
            *(f"batch_rack_tray_{tray:02d}_shelf_{shelf}_weld" for tray in range(1, 4) for shelf in range(3)),
        ]
        for name in required_bodies:
            try:
                model.body(name)
            except KeyError:
                errors.append(f"MJCF缺少body {name}")
        for name in required_sites:
            try:
                model.site(name)
            except KeyError:
                errors.append(f"MJCF缺少site {name}")
        for name in required_welds:
            try:
                model.equality(name)
            except KeyError:
                errors.append(f"MJCF缺少weld {name}")
        report = getattr(scene, "preflight_report", None)
        if report is not None and not report.ok:
            errors.append("现有运动/工装启动预检未通过")
    if errors:
        raise FlexiblePreflightError("柔性订单启动预检失败：\n- " + "\n- ".join(errors))


__all__ = ["FlexiblePreflightError", "validate_process_plan"]
