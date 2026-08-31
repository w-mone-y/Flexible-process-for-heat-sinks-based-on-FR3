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
    recovery_class: str = "AUTONOMOUS_RECOVERY"
    detection_stage: str = "运行时状态检测"
    recovery_route_zh: str = "安全停止后恢复原工序并复检"
    final_disposition_zh: str = "修复后复检，合格则回归原订单"
    physical_fault: str | None = None
    runtime_fault: str | None = None
    supports_duration: bool = False
    simulated_manual_review: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "label_zh": self.label_zh,
            "category_zh": self.category_zh,
            "target_kind": self.target_kind,
            "hint_zh": self.hint_zh,
            "recovery_class": self.recovery_class,
            "detection_stage": self.detection_stage,
            "recovery_route_zh": self.recovery_route_zh,
            "final_disposition_zh": self.final_disposition_zh,
            "physical_fault": self.physical_fault,
            "runtime_fault": self.runtime_fault,
            "supports_duration": self.supports_duration,
            "simulated_manual_review": self.simulated_manual_review,
        }


_DEFINITIONS = (
    ManualFaultDefinition(
        "FIN_POSE",
        "翅片位置/倾斜偏差",
        "物理质量故障",
        "fin",
        "目标翅片保持安装高度但横向错过梳齿槽；S4检出后原样返回S3B，由Arm3夹起、垂直抬升、纠偏回插并复检。",
        physical_fault="fin_pose",
        runtime_fault="FIN_GEOMETRY_FAILED",
        detection_stage="S4 焊前视觉检测",
        recovery_route_zh="原样返回 S3B → Arm3夹起偏移翅片 → 对正槽位慢速回插 → S4复检",
    ),
    ManualFaultDefinition(
        "BRAZING_MISSING",
        "钎料局部漏涂",
        "物理质量故障",
        "path",
        "指定黄色焊道的局部末段会真实缺失；相机检出后保留全部已涂焊道，Arm2只补该缺口并复检。",
        physical_fault="brazing_gap",
        runtime_fault="BRAZING_MISSING",
        detection_stage="S2B 钎料视觉检测",
        recovery_route_zh="保留现有焊道返回 S2A → 仅补涂缺口 → S2B复检",
    ),
    ManualFaultDefinition(
        "BRAZING_PATH_DEVIATION",
        "钎料轨迹偏移",
        "物理质量故障",
        "path",
        "目标黄色焊道会真实横向偏移；相机检出后仅纠正该焊道，其他排布保持不变并重新复检。",
        physical_fault="brazing_deviation",
        runtime_fault="BRAZING_PATH_DEVIATION",
        detection_stage="S2B 钎料视觉检测",
        recovery_route_zh="保留正常焊道返回 S2A → 渐进去除偏移焊道 → 原路径重涂 → S2B复检",
    ),
    ManualFaultDefinition(
        "FURNACE_PROFILE",
        "炉温曲线异常",
        "物理质量故障",
        "furnace",
        "下一热循环的热区异常变色；焊后相机检出后隔离产品并转人工工艺评审，不自动二次过炉。",
        physical_fault="furnace_profile",
        runtime_fault="FURNACE_PROFILE",
        recovery_class="MANUAL_DISPOSITION",
        detection_stage="焊后固定相机与炉温曲线复核",
        recovery_route_zh="隔离产品与炉批记录 → 10秒模拟人工工艺评审",
        final_disposition_zh="人工确认后解除停线；产品保持隔离记录",
        simulated_manual_review=True,
    ),
    ManualFaultDefinition(
        "FIN_PICK_FAILED",
        "翅片抓取失败",
        "机器人/工艺故障",
        "fin",
        "Arm3夹爪故意保留大于翅片厚度的间隙，目标翅片留在料台；S4检出缺片后托盘原样返回S3B，由Arm3只补装该片。",
        physical_fault="fin_pick",
        runtime_fault="FIN_PICK_FAILED",
        detection_stage="S4 焊前缺片检测",
        recovery_route_zh="保持其余产品状态返回 S3B → Arm3重新抓取目标翅片 → 单片慢速补装 → S4复检",
        final_disposition_zh="S4复检合格后自动回归原订单",
    ),
    ManualFaultDefinition(
        "FIN_GEOMETRY_FAILED",
        "翅片几何检测失败",
        "机器人/工艺故障",
        "fin",
        "目标翅片保持安装高度但横向错槽；S4检出后进入10秒人工审核并模拟修复。",
        physical_fault="fin_pose",
        runtime_fault="FIN_GEOMETRY_FAILED",
        recovery_class="MANUAL_DISPOSITION",
        detection_stage="S4 焊前几何检测",
        recovery_route_zh="隔离当前托盘 → 10秒模拟人工几何审核与修复",
        final_disposition_zh="人工确认修复后回归原订单",
        simulated_manual_review=True,
    ),
    ManualFaultDefinition(
        "ARM_UNAVAILABLE",
        "机械臂暂时离线",
        "设备故障",
        "arm",
        "指定机械臂停在当前姿态并整机变红；进入10秒人工审核，完成后解除隔离并继续原任务。",
        runtime_fault="ARM_UNAVAILABLE",
        recovery_class="MANUAL_DISPOSITION",
        detection_stage="机械臂心跳与执行反馈",
        recovery_route_zh="禁止新派工并保持安全姿态 → 10秒模拟人工检查 → 解除隔离",
        final_disposition_zh="资源恢复后从安全检查点继续原任务",
        simulated_manual_review=True,
    ),
    ManualFaultDefinition(
        "RACK_LAYER_UNAVAILABLE",
        "炉内料架层不可用",
        "物流设备故障",
        "rack_layer",
        "故障层的导轨、滚轮和锁销整层变红，待入架托盘自动改派其他空层。",
        runtime_fault="RACK_LAYER_UNAVAILABLE",
        detection_stage="入炉层位预留检查",
        recovery_route_zh="撤销未执行层位预留 → 原子改派其他空层 → 继续入炉",
    ),
    ManualFaultDefinition(
        "ELEVATOR_TIMEOUT",
        "炉前传送带驱动超时",
        "物流设备故障",
        "furnace_conveyor",
        "黑色入炉传送带停在当前位置并变红，恢复后继续直线输送。",
        runtime_fault="ELEVATOR_TIMEOUT",
        detection_stage="炉前移载位置与超时监控",
        recovery_route_zh="安全停机 → 托盘归属核验 → 升降机构回零 → 重试一次",
        supports_duration=True,
    ),
    ManualFaultDefinition(
        "FORK_TIMEOUT",
        "炉口托盘到位信号超时",
        "物流设备故障",
        "furnace_conveyor",
        "托盘在炉口停止并等待到位确认，恢复后重新校验归属并继续。",
        runtime_fault="FORK_TIMEOUT",
        detection_stage="炉口托盘到位信号监控",
        recovery_route_zh="保持托盘锁定 → 推叉回零 → 重新校验到位并重试一次",
        supports_duration=True,
    ),
    ManualFaultDefinition(
        "FURNACE_DOOR_INTERLOCK",
        "炉门互锁异常",
        "炉体故障",
        "furnace",
        "炉门会卡在半开位置且控制柜变红，禁止热循环并执行互锁复检。",
        runtime_fault="FURNACE_DOOR_INTERLOCK",
        detection_stage="炉门闭合与输送互锁检查",
        recovery_route_zh="禁止输送和热循环 → 保持料架锁定 → 恢复后重新执行互锁检查",
        supports_duration=True,
    ),
    ManualFaultDefinition(
        "CONTACT_SAFETY_STOP",
        "非预期接触安全停机",
        "安全故障",
        "current_task",
        "关联机构立即停止并变红，接触位置显示闪烁红球；默认需要人工确认。",
        runtime_fault="CONTACT_SAFETY_STOP",
        recovery_class="MANUAL_DISPOSITION",
        detection_stage="MuJoCo非预期接触与最小距离监控",
        recovery_route_zh="全单元安全保持 → 10秒模拟人工碰撞检查与复位",
        final_disposition_zh="人工确认安全后解除单元停机",
        supports_duration=False,
        simulated_manual_review=True,
    ),
    ManualFaultDefinition(
        "TRAY_STATE_INCONSISTENT",
        "托盘归属状态不一致",
        "安全故障",
        "current_task",
        "托盘变为紫红色并出现偏移半透明重影，停止物流并转入人工归属检查。",
        runtime_fault="TRAY_STATE_INCONSISTENT",
        recovery_class="MANUAL_DISPOSITION",
        detection_stage="托盘唯一所有权断言",
        recovery_route_zh="冻结相关物流 → 10秒模拟人工归属核验 → 重建唯一所有权",
        final_disposition_zh="所有权一致后恢复物流与原订单",
        supports_duration=False,
        simulated_manual_review=True,
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
    forced_manual = safety_fault or definition.simulated_manual_review
    auto_recover = False if forced_manual else bool(payload.get("auto_recover", True))
    duration = payload.get("duration_s", 8.0 if definition.supports_duration else None)
    if forced_manual:
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
