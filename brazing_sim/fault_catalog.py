"""User-facing manual fault catalogue shared by HTTP and the Qt console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ManualFaultDefinition:
    fault_type: str
    label_zh: str
    category_zh: str
    target_kind: str
    hint_zh: str
    physical_fault: str | None = None
    runtime_fault: str | None = None
    supports_duration: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "label_zh": self.label_zh,
            "category_zh": self.category_zh,
            "target_kind": self.target_kind,
            "hint_zh": self.hint_zh,
            "physical_fault": self.physical_fault,
            "runtime_fault": self.runtime_fault,
            "supports_duration": self.supports_duration,
        }


_DEFINITIONS = (
    ManualFaultDefinition(
        "FIN_POSE",
        "翅片位置/倾斜偏差",
        "物理质量故障",
        "fin",
        "翅片会在MuJoCo中真实平移并倾斜；Arm3检出后由Arm1调整并复检。",
        physical_fault="fin_pose",
        runtime_fault="FIN_GEOMETRY_FAILED",
    ),
    ManualFaultDefinition(
        "BRAZING_MISSING",
        "钎料局部漏涂",
        "物理质量故障",
        "path",
        "指定黄色焊道的局部末段会真实缺失；相机检出后保留全部已涂焊道，Arm2只补该缺口并复检。",
        physical_fault="brazing_gap",
        runtime_fault="BRAZING_MISSING",
    ),
    ManualFaultDefinition(
        "BRAZING_PATH_DEVIATION",
        "钎料轨迹偏移",
        "物理质量故障",
        "path",
        "目标黄色焊道会真实横向偏移；相机检出后仅纠正该焊道，其他排布保持不变并重新复检。",
        physical_fault="brazing_deviation",
        runtime_fault="BRAZING_PATH_DEVIATION",
    ),
    ManualFaultDefinition(
        "FURNACE_PROFILE",
        "炉温曲线异常",
        "物理质量故障",
        "furnace",
        "下一热循环的热区异常变色；焊后相机检出后隔离产品并转人工工艺评审，不自动二次过炉。",
        physical_fault="furnace_profile",
        runtime_fault="FURNACE_PROFILE",
    ),
    ManualFaultDefinition(
        "FIN_PICK_FAILED",
        "翅片抓取失败",
        "机器人/工艺故障",
        "fin",
        "目标翅片会明显留在Table1而不出现在槽内，Arm3检出后Arm1返回重抓。",
        physical_fault="fin_pick",
        runtime_fault="FIN_PICK_FAILED",
    ),
    ManualFaultDefinition(
        "FIN_INSERT_FAILED",
        "翅片插装失败",
        "机器人/工艺故障",
        "fin",
        "目标翅片会以抬高、偏位和倾斜姿态卡在槽口，Arm1重新插装并复检。",
        physical_fault="fin_insert",
        runtime_fault="FIN_INSERT_FAILED",
    ),
    ManualFaultDefinition(
        "FIN_GEOMETRY_FAILED",
        "翅片几何检测失败",
        "机器人/工艺故障",
        "fin",
        "等待翅片检测阶段触发，Arm1重新安装后由Arm3复检。",
        physical_fault="fin_pose",
        runtime_fault="FIN_GEOMETRY_FAILED",
    ),
    ManualFaultDefinition(
        "ARM_UNAVAILABLE",
        "机械臂暂时离线",
        "设备故障",
        "arm",
        "指定机械臂停在当前姿态并整机变红；可定时自动恢复或在资源页手动恢复。",
        runtime_fault="ARM_UNAVAILABLE",
        supports_duration=True,
    ),
    ManualFaultDefinition(
        "RACK_LAYER_UNAVAILABLE",
        "炉内料架层不可用",
        "物流设备故障",
        "rack_layer",
        "故障层的导轨、滚轮和锁销整层变红，待入架托盘自动改派其他空层。",
        runtime_fault="RACK_LAYER_UNAVAILABLE",
    ),
    ManualFaultDefinition(
        "ELEVATOR_TIMEOUT",
        "炉前传送带驱动超时",
        "物流设备故障",
        "furnace_conveyor",
        "黑色入炉传送带停在当前位置并变红，恢复后继续直线输送。",
        runtime_fault="ELEVATOR_TIMEOUT",
        supports_duration=True,
    ),
    ManualFaultDefinition(
        "FORK_TIMEOUT",
        "炉口托盘到位信号超时",
        "物流设备故障",
        "furnace_conveyor",
        "托盘在炉口停止并等待到位确认，恢复后重新校验归属并继续。",
        runtime_fault="FORK_TIMEOUT",
        supports_duration=True,
    ),
    ManualFaultDefinition(
        "FURNACE_DOOR_INTERLOCK",
        "炉门互锁异常",
        "炉体故障",
        "furnace",
        "炉门会卡在半开位置且控制柜变红，禁止热循环并执行互锁复检。",
        runtime_fault="FURNACE_DOOR_INTERLOCK",
        supports_duration=True,
    ),
    ManualFaultDefinition(
        "CONTACT_SAFETY_STOP",
        "非预期接触安全停机",
        "安全故障",
        "current_task",
        "关联机构立即停止并变红，接触位置显示闪烁红球；默认需要人工确认。",
        runtime_fault="CONTACT_SAFETY_STOP",
        supports_duration=False,
    ),
    ManualFaultDefinition(
        "TRAY_STATE_INCONSISTENT",
        "托盘归属状态不一致",
        "安全故障",
        "current_task",
        "托盘变为紫红色并出现偏移半透明重影，停止物流并转入人工归属检查。",
        runtime_fault="TRAY_STATE_INCONSISTENT",
        supports_duration=False,
    ),
)

MANUAL_FAULT_CATALOG = {definition.fault_type: definition for definition in _DEFINITIONS}


def fault_catalog_snapshot() -> list[dict[str, Any]]:
    return [definition.as_dict() for definition in _DEFINITIONS]


def validate_manual_fault_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    fault_type = str(payload.get("fault_type", "")).strip().upper()
    if fault_type not in MANUAL_FAULT_CATALOG:
        raise ValueError(f"unknown manual fault type: {fault_type or '-'}")
    definition = MANUAL_FAULT_CATALOG[fault_type]
    target = str(payload.get("target", "")).strip()
    if definition.target_kind in {"fin", "path", "arm", "rack_layer"} and not target:
        raise ValueError(f"{definition.label_zh} requires a target")
    if definition.target_kind == "fin" and not target.startswith("fin_"):
        raise ValueError("fin target must be fin_XX")
    # V1 names brazing paths ``slot_02_left``; V2 names them ``path_02``.  Both
    # are legitimate identifiers for the same kind of target, so accept either
    # rather than rejecting every path fault raised from the V2 console.
    if definition.target_kind == "path" and not target.startswith(("slot_", "path_")):
        raise ValueError("path target must be slot_XX_left/right or path_XX")
    if definition.target_kind == "arm":
        target = target.upper()
        if target not in {"ARM1", "ARM2", "ARM3"}:
            raise ValueError("arm target must be ARM1, ARM2 or ARM3")
    if definition.target_kind == "rack_layer":
        layer = int(target)
        if layer not in {0, 1, 2}:
            raise ValueError("rack layer must be 0, 1 or 2")
        target = str(layer)
    severity = str(payload.get("severity", "recoverable")).strip().lower()
    if severity not in {"recoverable", "severe"}:
        raise ValueError("severity must be recoverable or severe")
    safety_fault = fault_type in {"CONTACT_SAFETY_STOP", "TRAY_STATE_INCONSISTENT"}
    auto_recover = False if safety_fault else bool(payload.get("auto_recover", True))
    duration = payload.get("duration_s", 8.0 if definition.supports_duration else None)
    if safety_fault:
        duration = None
    if duration is not None:
        duration = float(duration)
        if not 0.5 <= duration <= 600.0:
            raise ValueError("duration_s must be between 0.5 and 600 seconds")
    return {
        "type": "manual_fault_inject",
        "fault_type": fault_type,
        "target": target,
        "severity": severity,
        "auto_recover": auto_recover,
        "duration_s": duration,
        "label_zh": definition.label_zh,
    }


__all__ = [
    "MANUAL_FAULT_CATALOG",
    "ManualFaultDefinition",
    "fault_catalog_snapshot",
    "validate_manual_fault_payload",
]
