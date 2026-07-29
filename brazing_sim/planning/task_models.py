"""Pure domain models used by planning and scheduling.

These objects intentionally do not reference MuJoCo model/data.  The execution
layer translates a :class:`ManufacturingTask` into the legacy ``TaskSpec``
command understood by the existing robot actors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RESERVED = "RESERVED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    RETRY_WAIT = "RETRY_WAIT"

    @property
    def terminal(self) -> bool:
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }


class TaskType(StrEnum):
    INDEX_MATERIAL_KIT = "INDEX_MATERIAL_KIT"
    INDEX_EMPTY_TRAY = "INDEX_EMPTY_TRAY"
    REMOVE_OLD_PRESS = "REMOVE_OLD_PRESS"
    REMOVE_OLD_COMB = "REMOVE_OLD_COMB"
    REMOVE_OLD_MOLD = "REMOVE_OLD_MOLD"
    FETCH_MOLD = "FETCH_MOLD"
    INSTALL_MOLD = "INSTALL_MOLD"
    VERIFY_MOLD = "VERIFY_MOLD"
    VERIFY_CHANGEOVER = "VERIFY_CHANGEOVER"
    PICK_BASE_PLATE = "PICK_BASE_PLATE"
    PLACE_BASE_PLATE = "PLACE_BASE_PLATE"
    VERIFY_BASE_ALIGNMENT = "VERIFY_BASE_ALIGNMENT"
    PREPARE_FIN_TOOL = "PREPARE_FIN_TOOL"
    CONFIGURE_COMB = "CONFIGURE_COMB"
    FETCH_COMB = "FETCH_COMB"
    INSTALL_COMB = "INSTALL_COMB"
    VERIFY_COMB = "VERIFY_COMB"
    DISPENSE_BRAZING = "DISPENSE_BRAZING"
    INSPECT_BRAZING = "INSPECT_BRAZING"
    REWORK_BRAZING = "REWORK_BRAZING"
    PICK_FIN = "PICK_FIN"
    INSTALL_FIN = "INSTALL_FIN"
    INSPECT_FINS = "INSPECT_FINS"
    REINSTALL_FIN = "REINSTALL_FIN"
    FETCH_PRESS_MODULE = "FETCH_PRESS_MODULE"
    INSTALL_PRESS_MODULE = "INSTALL_PRESS_MODULE"
    APPLY_PRESS = "APPLY_PRESS"
    LOCK_FIXTURE = "LOCK_FIXTURE"
    TRANSFER_S1_S2A = "TRANSFER_S1_S2A"
    TRANSFER_S2A_S2B = "TRANSFER_S2A_S2B"
    TRANSFER_S2B_S3 = "TRANSFER_S2B_S3"
    TRANSFER_S3_RACK = "TRANSFER_S3_RACK"
    VERIFY_TRANSFER = "VERIFY_TRANSFER"
    # Retained only for loading historical experiment snapshots.  New plans
    # use the four independent asynchronous transfer tasks above.
    ROTATE_TABLE2 = "ROTATE_TABLE2"
    VERIFY_TURNTABLE = "VERIFY_TURNTABLE"
    TRANSFER_TRAY_OUT = "TRANSFER_TRAY_OUT"
    MOVE_ELEVATOR = "MOVE_ELEVATOR"
    LOAD_RACK_LAYER = "LOAD_RACK_LAYER"
    LOCK_RACK_LAYER = "LOCK_RACK_LAYER"
    BATCH_READY = "BATCH_READY"
    RUN_FURNACE = "RUN_FURNACE"
    UNLOAD_RACK_LAYER = "UNLOAD_RACK_LAYER"
    POST_BRAZE_INSPECTION = "POST_BRAZE_INSPECTION"
    SECOND_POST_BRAZE_VIEW = "SECOND_POST_BRAZE_VIEW"
    ROUTE_PASS = "ROUTE_PASS"
    ROUTE_REWORK = "ROUTE_REWORK"
    ROUTE_SCRAP = "ROUTE_SCRAP"
    SAFE_HOME_TRANSFER = "SAFE_HOME_TRANSFER"
    FURNACE_INTERLOCK_CHECK = "FURNACE_INTERLOCK_CHECK"


TASK_TYPE_LABELS_ZH: dict[TaskType, str] = {
    TaskType.INDEX_MATERIAL_KIT: "索引订单物料盒",
    TaskType.INDEX_EMPTY_TRAY: "空托盘补位",
    TaskType.REMOVE_OLD_PRESS: "拆除双压片",
    TaskType.REMOVE_OLD_COMB: "拆除旧梳齿",
    TaskType.REMOVE_OLD_MOLD: "拆除旧托盘模具",
    TaskType.FETCH_MOLD: "龙门取出目标模具",
    TaskType.INSTALL_MOLD: "安装低矮托盘模具",
    TaskType.VERIFY_MOLD: "验证模具定位锁销",
    TaskType.VERIFY_CHANGEOVER: "高可靠换型复核",
    TaskType.PICK_BASE_PLATE: "吸取基板",
    TaskType.PLACE_BASE_PLATE: "搬运并放置基板",
    TaskType.VERIFY_BASE_ALIGNMENT: "复核基板定位",
    TaskType.PREPARE_FIN_TOOL: "切换翅片夹爪",
    TaskType.CONFIGURE_COMB: "安装梳齿夹具",
    TaskType.FETCH_COMB: "龙门取出梳齿模块",
    TaskType.INSTALL_COMB: "安装前后梳齿模块",
    TaskType.VERIFY_COMB: "验证梳齿模块锁定",
    TaskType.DISPENSE_BRAZING: "涂覆钎料",
    TaskType.INSPECT_BRAZING: "检测钎料涂覆",
    TaskType.REWORK_BRAZING: "局部补涂钎料",
    TaskType.PICK_FIN: "夹取翅片",
    TaskType.INSTALL_FIN: "安装翅片",
    TaskType.INSPECT_FINS: "检测翅片安装",
    TaskType.REINSTALL_FIN: "重新安装翅片",
    TaskType.FETCH_PRESS_MODULE: "龙门取出短压梁",
    TaskType.INSTALL_PRESS_MODULE: "安装两根短压梁",
    TaskType.APPLY_PRESS: "压梁压紧翅片",
    TaskType.LOCK_FIXTURE: "锁紧托盘夹具",
    TaskType.TRANSFER_S1_S2A: "托盘移载：S1→S2A",
    TaskType.TRANSFER_S2A_S2B: "托盘移载：S2A→S2B",
    TaskType.TRANSFER_S2B_S3: "托盘移载：S2B→S3",
    TaskType.TRANSFER_S3_RACK: "托盘移载：S3→炉前料架",
    TaskType.VERIFY_TRANSFER: "确认托盘移载到位",
    TaskType.ROTATE_TABLE2: "双巢位转台换位",
    TaskType.VERIFY_TURNTABLE: "确认转台停稳对位",
    TaskType.TRANSFER_TRAY_OUT: "托盘进入炉前传送带",
    TaskType.MOVE_ELEVATOR: "炉内层位分配",
    TaskType.LOAD_RACK_LAYER: "直线传送带送入料架",
    TaskType.LOCK_RACK_LAYER: "锁定炉内料架层",
    TaskType.BATCH_READY: "确认炉批装载完成",
    TaskType.RUN_FURNACE: "执行炉内钎焊",
    TaskType.UNLOAD_RACK_LAYER: "从料架卸载托盘",
    TaskType.POST_BRAZE_INSPECTION: "执行焊后检测",
    TaskType.SECOND_POST_BRAZE_VIEW: "焊后第二视角复检",
    TaskType.ROUTE_PASS: "合格品分流",
    TaskType.ROUTE_REWORK: "返工品分流",
    TaskType.ROUTE_SCRAP: "报废品分流",
    TaskType.SAFE_HOME_TRANSFER: "移载机构安全回零",
    TaskType.FURNACE_INTERLOCK_CHECK: "重新检查炉门互锁",
}

TASK_STATUS_LABELS_ZH: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "待执行",
    TaskStatus.READY: "可以执行",
    TaskStatus.RESERVED: "资源已预留",
    TaskStatus.RUNNING: "正在执行",
    TaskStatus.SUCCEEDED: "已完成",
    TaskStatus.FAILED: "执行失败",
    TaskStatus.BLOCKED: "已阻塞",
    TaskStatus.CANCELLED: "已取消",
    TaskStatus.RETRY_WAIT: "等待重试",
}


def task_type_label_zh(value: TaskType | str) -> str:
    """Return the authoritative Chinese process name for a task type."""

    try:
        task_type = TaskType(value)
    except (TypeError, ValueError):
        return "未知任务"
    return TASK_TYPE_LABELS_ZH[task_type]


def task_status_label_zh(value: TaskStatus | str) -> str:
    """Return a Chinese runtime status without leaking enum identifiers."""

    try:
        status = TaskStatus(value)
    except (TypeError, ValueError):
        return "状态未知"
    return TASK_STATUS_LABELS_ZH[status]


def _number_suffix(value: object) -> str:
    digits = "".join(character for character in str(value).rsplit("_", 1)[-1] if character.isdigit())
    return str(int(digits)) if digits else ""


def _unit_label_zh(unit_id: object) -> str:
    value = str(unit_id)
    if value.endswith("_BATCH") or value == "SYSTEM":
        return "整炉批次" if value.endswith("_BATCH") else "系统恢复"
    number = _number_suffix(value)
    return f"工件{number}" if number else "当前工件"


def _path_label_zh(path_id: object) -> str:
    value = str(path_id)
    side = "左侧" if value.endswith("_left") else "右侧" if value.endswith("_right") else ""
    stem = value.removesuffix("_left").removesuffix("_right")
    number = _number_suffix(stem)
    return f"翅片{number}{side}路径" if number else "指定钎料路径"


def task_detail_label_zh(task: Mapping[str, Any]) -> str:
    """Build a payload-aware Chinese detail so repeated nodes cannot be confused."""

    payload = task.get("payload", {})
    payload = payload if isinstance(payload, Mapping) else {}
    task_type_value = task.get("task_type", "")
    try:
        task_type = TaskType(task_type_value)
    except (TypeError, ValueError):
        task_type = None
    unit = _unit_label_zh(task.get("unit_id", ""))
    detail = ""
    if task_type in {TaskType.PICK_FIN, TaskType.INSTALL_FIN, TaskType.REINSTALL_FIN}:
        number = _number_suffix(payload.get("fin_id", ""))
        detail = f"翅片{number}" if number else "指定翅片"
    elif task_type in {
        TaskType.DISPENSE_BRAZING,
        TaskType.INSPECT_BRAZING,
        TaskType.REWORK_BRAZING,
    }:
        path_ids = payload.get("path_ids", ())
        if isinstance(path_ids, (list, tuple)) and len(path_ids) == 1:
            detail = _path_label_zh(path_ids[0])
        elif isinstance(path_ids, (list, tuple)) and path_ids:
            detail = f"共{len(path_ids)}条钎料路径"
        else:
            detail = "钎料路径"
    elif task_type is TaskType.CONFIGURE_COMB:
        module = str(payload.get("comb_module_name", ""))
        spacing = "".join(character for character in module if character.isdigit())
        detail = f"{spacing}毫米梳齿模块" if spacing else "订单匹配梳齿模块"
    elif task_type is TaskType.REMOVE_OLD_COMB and payload.get("after_brazing"):
        detail = "焊后梳齿板缓慢退出"
    elif task_type is TaskType.REMOVE_OLD_PRESS and payload.get("after_brazing"):
        detail = "两根短压片同步抬升并退出"
    elif task_type in {
        TaskType.FETCH_COMB,
        TaskType.INSTALL_COMB,
        TaskType.VERIFY_COMB,
        TaskType.FETCH_MOLD,
        TaskType.INSTALL_MOLD,
        TaskType.VERIFY_MOLD,
        TaskType.VERIFY_CHANGEOVER,
    }:
        module = str(payload.get("module_name", payload.get("comb_module_name", "")))
        detail = module or "订单匹配实体模块"
    elif task_type in {
        TaskType.TRANSFER_S1_S2A,
        TaskType.TRANSFER_S2A_S2B,
        TaskType.TRANSFER_S2B_S3,
        TaskType.TRANSFER_S3_RACK,
        TaskType.VERIFY_TRANSFER,
    }:
        source = str(payload.get("source_station", ""))
        target = str(payload.get("target_station", ""))
        detail = f"{source}→{target}" if source and target else "独立短行程滑台"
    elif task_type in {TaskType.ROTATE_TABLE2, TaskType.VERIFY_TURNTABLE}:
        nest = str(task.get("nest_id") or "双巢位")
        detail = f"{nest} · 180度同步换位"
    elif task_type is TaskType.VERIFY_BASE_ALIGNMENT:
        detail = "高可靠路线 · 基板位姿二次确认"
    elif task_type is TaskType.SECOND_POST_BRAZE_VIEW:
        detail = "高可靠路线 · 侧视检测"
    elif task_type is TaskType.APPLY_PRESS:
        force = payload.get("target_force_n")
        detail = f"目标夹紧力{float(force):g}牛" if isinstance(force, (int, float)) else "订单目标夹紧力"
    elif task_type in {
        TaskType.MOVE_ELEVATOR,
        TaskType.LOAD_RACK_LAYER,
        TaskType.LOCK_RACK_LAYER,
        TaskType.UNLOAD_RACK_LAYER,
    }:
        layer = payload.get("layer_index")
        detail = f"第{int(layer) + 1}层" if isinstance(layer, int) else "计划层位"
    elif task_type is TaskType.RUN_FURNACE:
        duration = payload.get("duration_s")
        detail = f"钎焊周期{float(duration):g}秒" if isinstance(duration, (int, float)) else "整炉热循环"
    elif task_type is TaskType.BATCH_READY:
        detail = "全部计划托盘已锁定"
    elif task_type in {TaskType.ROUTE_PASS, TaskType.ROUTE_REWORK, TaskType.ROUTE_SCRAP}:
        detail = "按焊后检测结果执行"
    elif task_type is TaskType.SAFE_HOME_TRANSFER:
        detail = "故障恢复任务"
    elif task_type is TaskType.FURNACE_INTERLOCK_CHECK:
        detail = "安全恢复任务"
    if task.get("recovery_for") and "恢复" not in detail:
        detail = f"恢复任务 · {detail or unit}"
    elif (
        detail
        and unit not in {"整炉批次", "系统恢复"}
        and task_type not in {TaskType.BATCH_READY, TaskType.RUN_FURNACE}
    ):
        detail = f"{unit} · {detail}"
    elif not detail:
        detail = unit
    return detail


@dataclass(slots=True)
class ManufacturingTask:
    task_id: str
    task_type: TaskType | str
    order_id: str
    unit_id: str
    tray_id: str | None = None
    station_id: str | None = None
    nest_id: str | None = None
    station_capabilities: list[str] = field(default_factory=list)
    route_phase: str | None = None
    motion_constraints: dict[str, Any] = field(default_factory=dict)
    reservation_id: str | None = None
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    eligible_resources: list[str] = field(default_factory=list)
    required_tool: str | None = None
    required_zones: list[str] = field(default_factory=list)
    estimated_duration: float = 0.1
    priority: int = 0
    retry_limit: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus | str = TaskStatus.PENDING
    assigned_resource: str | None = None
    retry_count: int = 0
    created_at: float = 0.0
    ready_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    failure_reason: str | None = None
    recovery_for: str | None = None
    sequence_index: int = 0

    def __post_init__(self) -> None:
        self.task_type = TaskType(self.task_type)
        self.status = TaskStatus(self.status)
        if not self.task_id or not self.order_id or not self.unit_id:
            raise ValueError("task_id, order_id and unit_id must not be empty")
        self.predecessors = list(dict.fromkeys(str(value) for value in self.predecessors))
        self.successors = list(dict.fromkeys(str(value) for value in self.successors))
        self.eligible_resources = list(dict.fromkeys(str(value).upper() for value in self.eligible_resources))
        self.required_zones = list(dict.fromkeys(str(value).upper() for value in self.required_zones))
        self.station_capabilities = list(
            dict.fromkeys(str(value).upper() for value in self.station_capabilities)
        )
        if self.station_id is not None:
            self.station_id = str(self.station_id).upper()
        if self.nest_id is not None:
            self.nest_id = str(self.nest_id).upper()
        if self.route_phase is not None:
            self.route_phase = str(self.route_phase).upper()
        if not isfinite(self.estimated_duration) or self.estimated_duration < 0:
            raise ValueError("estimated_duration must be finite and non-negative")
        if self.retry_limit < 0 or self.retry_count < 0:
            raise ValueError("retry counts must be non-negative")

    @property
    def is_recovery(self) -> bool:
        return self.recovery_for is not None

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.retry_limit

    def mark_ready(self, now: float) -> None:
        if self.status in {TaskStatus.PENDING, TaskStatus.RETRY_WAIT}:
            self.status = TaskStatus.READY
            if self.ready_at is None:
                self.ready_at = float(now)

    def reserve(self, resource_id: str) -> None:
        if self.status is not TaskStatus.READY:
            raise RuntimeError(f"task {self.task_id} is not READY")
        resource = str(resource_id).upper()
        if self.eligible_resources and resource not in self.eligible_resources:
            raise ValueError(f"resource {resource} is not eligible for {self.task_id}")
        self.assigned_resource = resource
        self.status = TaskStatus.RESERVED

    def mark_running(self, now: float) -> None:
        if self.status is not TaskStatus.RESERVED:
            raise RuntimeError(f"task {self.task_id} is not RESERVED")
        self.status = TaskStatus.RUNNING
        self.started_at = float(now)

    def mark_succeeded(self, now: float) -> None:
        if self.status.terminal:
            return
        self.status = TaskStatus.SUCCEEDED
        self.finished_at = float(now)

    def mark_failed(self, now: float, reason: str) -> None:
        if self.status.terminal:
            return
        self.status = TaskStatus.FAILED
        self.finished_at = float(now)
        self.failure_reason = str(reason)

    def prepare_retry(self, now: float) -> bool:
        if self.status is not TaskStatus.FAILED or not self.can_retry:
            return False
        self.retry_count += 1
        self.status = TaskStatus.RETRY_WAIT
        self.assigned_resource = None
        self.started_at = None
        self.finished_at = None
        self.failure_reason = None
        self.ready_at = float(now)
        return True

    def as_dict(self) -> dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "order_id": self.order_id,
            "unit_id": self.unit_id,
            "tray_id": self.tray_id,
            "station_id": self.station_id,
            "nest_id": self.nest_id,
            "station_capabilities": list(self.station_capabilities),
            "route_phase": self.route_phase,
            "motion_constraints": dict(self.motion_constraints),
            "reservation_id": self.reservation_id,
            "predecessors": list(self.predecessors),
            "successors": list(self.successors),
            "eligible_resources": list(self.eligible_resources),
            "required_tool": self.required_tool,
            "required_zones": list(self.required_zones),
            "estimated_duration": self.estimated_duration,
            "priority": self.priority,
            "retry_limit": self.retry_limit,
            "payload": dict(self.payload),
            "status": self.status.value,
            "assigned_resource": self.assigned_resource,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "ready_at": self.ready_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "failure_reason": self.failure_reason,
            "recovery_for": self.recovery_for,
            "sequence_index": self.sequence_index,
        }
        result["display_name_zh"] = task_type_label_zh(self.task_type)
        result["display_detail_zh"] = task_detail_label_zh(result)
        result["status_zh"] = task_status_label_zh(self.status)
        return result


__all__ = [
    "ManufacturingTask",
    "TASK_STATUS_LABELS_ZH",
    "TASK_TYPE_LABELS_ZH",
    "TaskStatus",
    "TaskType",
    "task_detail_label_zh",
    "task_status_label_zh",
    "task_type_label_zh",
]
