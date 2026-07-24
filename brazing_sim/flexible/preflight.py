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
            "station_s1_anchor",
            "station_s2a_anchor",
            "station_s2b_anchor",
            "station_s3_anchor",
            "station_rack_infeed_anchor",
            "transfer_s1_s2a_carriage",
            "transfer_s2a_s2b_carriage",
            "transfer_s2b_s3_carriage",
            "transfer_s3_rack_carriage",
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
            "s1_target_site",
            "s2a_target_site",
            "s2b_target_site",
            "s3_target_site",
            "rack_infeed_target_site",
            *(f"fin_slot_{index:02d}_target" for index in range(1, 13)),
            *(f"raw_fin_{index:02d}_site" for index in range(1, 13)),
            "arm2_dispenser_center_tcp",
            "arm2_left_nozzle_tip_site",
            "arm2_right_nozzle_tip_site",
        ]
        required_geoms = [
            *(
                f"batch_tray_{unit:02d}_{part}"
                for unit in range(1, 4)
                for part in (
                    "template_plate",
                    "front_comb_base",
                    "rear_comb_base",
                )
            ),
        ]
        required_welds = [
            *(f"batch_rack_tray_{tray:02d}_shelf_{shelf}_weld" for tray in range(1, 4) for shelf in range(3)),
        ]
        required_joints = [
            "transfer_s1_s2a_joint",
            "transfer_s2a_s2b_joint",
            "transfer_s2b_s3_joint",
            "transfer_s3_rack_joint",
        ]
        required_actuators = [
            "transfer_s1_s2a_actuator",
            "transfer_s2a_s2b_actuator",
            "transfer_s2b_s3_actuator",
            "transfer_s3_rack_actuator",
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
        for name in required_geoms:
            try:
                model.geom(name)
            except KeyError:
                errors.append(f"MJCF缺少geom {name}")
        for name in required_welds:
            try:
                model.equality(name)
            except KeyError:
                errors.append(f"MJCF缺少weld {name}")
        for name in required_joints:
            try:
                model.joint(name)
            except KeyError:
                errors.append(f"MJCF缺少joint {name}")
        for name in required_actuators:
            try:
                model.actuator(name)
            except KeyError:
                errors.append(f"MJCF缺少actuator {name}")
        report = getattr(scene, "preflight_report", None)
        if report is not None and not report.ok:
            errors.append("现有运动/工装启动预检未通过")
    if errors:
        raise FlexiblePreflightError("柔性订单启动预检失败：\n- " + "\n- ".join(errors))


__all__ = ["FlexiblePreflightError", "validate_process_plan"]
