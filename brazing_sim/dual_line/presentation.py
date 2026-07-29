"""Adapt V2 runtime truth to the shared V1 HTTP/Qt state contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from brazing_sim.ui import line_ui_profile

_OWNER_TO_STATION = {
    "S1": "S1_BASE_LOADING",
    "S2A": "S2A_DISPENSING",
    "S2B": "S2B_MATERIAL_INSPECTION",
    "INSTALL_A": "S3A_ARM1_INSTALL",
    "INSTALL_B": "S3B_ARM3_INSTALL",
    "MERGE_A_WAIT": "MERGE_A_WAIT",
    "MERGE_B_WAIT": "MERGE_B_WAIT",
    "MERGE": "Y_MERGE_SHARED",
    "S4": "S4_PRE_BRAZE_INSPECTION",
    "BUFFER_1": "FURNACE_BUFFER_1",
    "BUFFER_2": "FURNACE_BUFFER_2",
    "BUFFER_3": "FURNACE_BUFFER_3",
    "POST_SCAN": "POST_BRAZE_SCAN",
    "OUTPUT": "FINISHED_OUTPUT",
}

_FURNACE_TEMPERATURE_C = {
    "IDLE": 25.0,
    "PLANNED": 25.0,
    "LOADING": 25.0,
    "PREHEAT": 180.0,
    "RAMP": 430.0,
    "SOAK": 600.0,
    "COOLING": 220.0,
    "READY_TO_UNLOAD": 80.0,
    "UNLOADING": 70.0,
    "COMPLETE": 25.0,
}

_STAGE_ORDER = (
    "QUEUED",
    "BASE_LOADING",
    "WAITING_S2A",
    "DISPENSING",
    "WAITING_S2B",
    "MATERIAL_INSPECTION",
    "WAITING_INSTALL",
    "FIN_INSTALLATION",
    "WAITING_MERGE",
    "MERGING",
    "WAITING_S4",
    "PRE_BRAZE_INSPECTION",
    "WAITING_BUFFER",
    "FURNACE_BUFFER",
    "FURNACE_LOADING",
    "BRAZING",
    "FURNACE_UNLOADING",
    "POST_BRAZE_INSPECTION",
    "WAITING_OUTPUT",
    "DELIVERING",
    "PRODUCT_REMOVED",
    "VIRTUAL_RETURN",
    "COMPLETE",
)
_STAGE_RANK = {stage: index for index, stage in enumerate(_STAGE_ORDER)}
_STATUS_ZH = {
    "PENDING": "等待前置任务",
    "READY": "已就绪",
    "RUNNING": "执行中",
    "SUCCEEDED": "已完成",
}


@dataclass(slots=True)
class V2StatePresenter:
    """One-way projection; presentation code never mutates runtime state."""

    line_profile: str = "V2_DUAL_INSTALL"

    @staticmethod
    def _active_unit(units: list[dict[str, Any]]) -> dict[str, Any] | None:
        return next((unit for unit in units if unit.get("stage") != "COMPLETE"), None)

    @staticmethod
    def _arm_states(operations: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        arms: dict[str, dict[str, Any]] = {}
        for resource in ("ARM1", "ARM2", "ARM3"):
            operation = operations.get(resource)
            arms[resource.lower()] = {
                "task_id": "" if operation is None else str(operation.get("unit_id", "")),
                "task_type": "" if operation is None else str(operation.get("kind", "")),
                "status": "idle" if operation is None else "running",
                "error": "",
            }
        return arms

    @staticmethod
    def _workstations(
        topology: Mapping[str, Any],
        trays: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        occupants = {
            _OWNER_TO_STATION.get(str(tray.get("owner"))): tray
            for tray in trays
            if _OWNER_TO_STATION.get(str(tray.get("owner"))) is not None
        }
        result: dict[str, dict[str, Any]] = {}
        for station in topology.get("stations", []):
            station_id = str(station.get("station_id", ""))
            tray = occupants.get(station_id)
            result[station_id] = {
                "station_id": station_id,
                "label_zh": station.get("label_zh", station_id),
                "world_xyz": list(station.get("world_xyz", ())),
                "tray_id": None if tray is None else tray.get("tray_id"),
                "occupied_by": None if tray is None else tray.get("unit_id"),
                "safe_for_transfer": tray is None,
            }
        return result

    @staticmethod
    def _orders(
        orders: list[dict[str, Any]],
        units: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        units_by_id = {str(unit.get("unit_id")): unit for unit in units}
        presented: list[dict[str, Any]] = []
        for order in orders:
            unit_ids = [str(value) for value in order.get("unit_ids", ())]
            complete_count = sum(
                units_by_id.get(unit_id, {}).get("stage") == "COMPLETE" for unit_id in unit_ids
            )
            quantity = len(unit_ids)
            presented.append(
                {
                    **order,
                    "quantity": quantity,
                    "status": "COMPLETE" if complete_count == quantity and quantity else "RUNNING",
                    "progress": complete_count / quantity if quantity else 0.0,
                    "urgent": bool(order.get("urgent", False)),
                }
            )
        return presented

    @staticmethod
    def _task_status(
        *,
        completed: bool,
        running: bool,
        ready: bool,
    ) -> str:
        if completed:
            return "SUCCEEDED"
        if running:
            return "RUNNING"
        return "READY" if ready else "PENDING"

    @classmethod
    def _tasks(
        cls,
        units: list[dict[str, Any]],
        operations: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        active_operations = {
            (
                str(operation.get("unit_id", "")),
                str(operation.get("kind", "")),
            ): resource
            for resource, operation in operations.items()
            if isinstance(operation, Mapping)
        }
        tasks: list[dict[str, Any]] = []
        for unit in units:
            unit_id = str(unit.get("unit_id", ""))
            tray_id = unit.get("tray_id")
            stage = str(unit.get("stage", "QUEUED"))
            rank = _STAGE_RANK.get(stage, 0)
            branch = str(unit.get("branch") or "")
            fin_count = int(unit.get("fin_count", 0))
            fins_installed = int(unit.get("fins_installed", 0))
            previous: str | None = None

            def append_task(
                suffix: str,
                task_type: str,
                title: str,
                station_id: str,
                *,
                completed: bool,
                running_kind: str,
                ready: bool,
                resource: str,
                detail: str = "",
            ) -> None:
                nonlocal previous
                task_id = f"{unit_id}:{suffix}"
                running_resource = active_operations.get((unit_id, running_kind))
                status = cls._task_status(
                    completed=completed,
                    running=running_resource is not None,
                    ready=ready,
                )
                tasks.append(
                    {
                        "task_id": task_id,
                        "unit_id": unit_id,
                        "tray_id": tray_id,
                        "task_type": task_type,
                        "operation_kind": running_kind,
                        "display_name_zh": title,
                        "display_detail_zh": detail or f"{tray_id or '待分配托盘'} · {station_id}",
                        "status": status,
                        "status_zh": _STATUS_ZH[status],
                        "predecessors": [] if previous is None else [previous],
                        "assigned_resource": running_resource
                        or (resource if status == "SUCCEEDED" else None),
                        "eligible_resources": [resource],
                        "station_id": station_id,
                        "required_zones": [station_id],
                        "failure_reason": "",
                    }
                )
                previous = task_id

            append_task(
                "base",
                "LOAD_BASE",
                "吸取并定位基板",
                "S1_BASE_LOADING",
                completed=rank > _STAGE_RANK["BASE_LOADING"],
                running_kind="BASE_LOADING",
                ready=rank >= _STAGE_RANK["BASE_LOADING"],
                resource="ARM1",
            )
            append_task(
                "dispense",
                "DISPENSE_BRAZING",
                "双喷嘴连续涂覆",
                "S2A_DISPENSING",
                completed=rank > _STAGE_RANK["DISPENSING"],
                running_kind="DISPENSING",
                ready=rank >= _STAGE_RANK["DISPENSING"],
                resource="ARM2",
            )
            append_task(
                "material_inspection",
                "INSPECT_BRAZING",
                "焊料质量检测与分流",
                "S2B_MATERIAL_INSPECTION",
                completed=rank > _STAGE_RANK["MATERIAL_INSPECTION"],
                running_kind="MATERIAL_INSPECTION",
                ready=rank >= _STAGE_RANK["MATERIAL_INSPECTION"],
                resource="ARM3",
            )
            if branch == "ARM1_A":
                install_station = "S3A_ARM1_INSTALL"
                install_resource = "ARM1"
            elif branch == "ARM3_B":
                install_station = "S3B_ARM3_INSTALL"
                install_resource = "ARM3"
            else:
                install_station = "INSTALL_BRANCH_PENDING"
                install_resource = "ARM1_OR_ARM3"
            for index in range(1, fin_count + 1):
                append_task(
                    f"fin_{index:02d}",
                    "INSTALL_FIN",
                    f"安装第 {index} / {fin_count} 片翅片",
                    install_station,
                    completed=fins_installed >= index,
                    running_kind="INSTALL_FIN" if fins_installed + 1 == index else "",
                    ready=(rank >= _STAGE_RANK["FIN_INSTALLATION"] and fins_installed + 1 >= index),
                    resource=install_resource,
                    detail=f"{tray_id or '待分配托盘'} · {branch or '等待支路分配'}",
                )
            append_task(
                "merge",
                "MERGE_BRANCH",
                "Y 形单占用合流",
                "Y_MERGE_SHARED",
                completed=rank > _STAGE_RANK["MERGING"],
                running_kind="MERGING",
                ready=rank >= _STAGE_RANK["WAITING_MERGE"],
                resource="MERGE",
            )
            append_task(
                "pre_braze",
                "PRE_BRAZE_INSPECTION",
                "共享焊前检测",
                "S4_PRE_BRAZE_INSPECTION",
                completed=rank > _STAGE_RANK["PRE_BRAZE_INSPECTION"],
                running_kind="PRE_BRAZE_INSPECTION",
                ready=rank >= _STAGE_RANK["PRE_BRAZE_INSPECTION"],
                resource="ARM3",
            )
            append_task(
                "buffer",
                "BUFFER_TRAY",
                "进入三位炉前缓存",
                "FURNACE_BUFFER",
                completed=rank >= _STAGE_RANK["FURNACE_BUFFER"],
                running_kind="",
                ready=rank >= _STAGE_RANK["WAITING_BUFFER"],
                resource="BUFFER_INDEX",
            )
            append_task(
                "furnace_load",
                "LOAD_FURNACE",
                "升降对层并由推叉装炉",
                "FURNACE_FRONT",
                completed=rank > _STAGE_RANK["FURNACE_LOADING"],
                running_kind="FURNACE_LOAD_TRAY",
                ready=rank >= _STAGE_RANK["FURNACE_LOADING"],
                resource="FURNACE_TRANSFER",
            )
            append_task(
                "brazing",
                "RUN_FURNACE",
                "三层兼容批次 CAB 钎焊",
                "FURNACE",
                completed=rank > _STAGE_RANK["BRAZING"],
                running_kind="",
                ready=rank >= _STAGE_RANK["BRAZING"],
                resource="FURNACE",
            )
            append_task(
                "furnace_unload",
                "UNLOAD_FURNACE",
                "后拉叉退架并下降",
                "FURNACE_REAR",
                completed=rank > _STAGE_RANK["FURNACE_UNLOADING"],
                running_kind="FURNACE_UNLOAD_TRAY",
                ready=rank >= _STAGE_RANK["FURNACE_UNLOADING"],
                resource="FURNACE_TRANSFER",
            )
            append_task(
                "post_braze",
                "POST_BRAZE_INSPECTION",
                "固定相机焊后检测",
                "POST_BRAZE_SCAN",
                completed=rank > _STAGE_RANK["POST_BRAZE_INSPECTION"],
                running_kind="POST_BRAZE_INSPECTION",
                ready=rank >= _STAGE_RANK["POST_BRAZE_INSPECTION"],
                resource="POST_CAMERA",
            )
            append_task(
                "delivery",
                "DELIVER_PRODUCT",
                "托盘送入成品出口并虚拟回流",
                "FINISHED_OUTPUT",
                completed=stage == "COMPLETE",
                running_kind="OUTPUT_DELIVERY",
                ready=rank >= _STAGE_RANK["WAITING_OUTPUT"],
                resource="OUTPUT",
            )
        return tasks

    @staticmethod
    def _resources(
        operations: Mapping[str, Any],
        tasks: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        tasks_by_operation = {
            (
                str(task.get("unit_id", "")),
                str(task.get("operation_kind", "")),
            ): str(task.get("task_id", ""))
            for task in tasks
        }
        tools = {
            "ARM1": "吸盘 / 平行夹爪",
            "ARM2": "固定双喷嘴焊料枪",
            "ARM3": "相机 + 轻型夹爪复合末端",
            "MERGE": "Y形合流器",
            "FURNACE_TRANSFER": "升降台 + 前后推拉叉",
            "FURNACE": "三层贯通式 CAB 炉",
            "POST_CAMERA": "固定焊后相机",
            "OUTPUT": "成品出口输送",
        }
        result: dict[str, dict[str, Any]] = {}
        for resource, tool in tools.items():
            operation = operations.get(resource)
            unit_id = "" if operation is None else str(operation.get("unit_id", ""))
            kind = "" if operation is None else str(operation.get("kind", ""))
            result[resource] = {
                "resource_type": "ROBOT" if resource.startswith("ARM") else "EQUIPMENT",
                "status": "BUSY" if operation is not None else "IDLE",
                "current_task_id": tasks_by_operation.get((unit_id, kind)),
                "current_tool": tool,
                "fault_code": None,
                "occupied_zones": [],
            }
        return result

    def present(
        self,
        snapshot: Mapping[str, Any],
        *,
        simulation_speed: float,
        actual_rtf: float,
        viewer_running: bool = False,
    ) -> dict[str, Any]:
        units = [dict(unit) for unit in snapshot.get("units", ())]
        trays = [dict(tray) for tray in snapshot.get("trays", ())]
        operations = dict(snapshot.get("operations", {}))
        active = self._active_unit(units)
        paused = bool(snapshot.get("paused", False))
        complete = bool(snapshot.get("complete", False))
        if paused:
            status = "paused"
        elif complete:
            status = "complete"
        elif units:
            status = "running"
        else:
            status = "idle"
        stage = "IDLE" if active is None else str(active.get("stage", "IDLE"))
        furnace = dict(snapshot.get("furnace", {}))
        furnace_phase = str(furnace.get("phase", "IDLE"))
        arm_states = self._arm_states(operations)
        tasks = self._tasks(units, operations)
        resources_v2 = self._resources(operations, tasks)
        profile = line_ui_profile(self.line_profile)
        segment_capabilities = {action.segment: False for action in profile.segment_actions}
        rack_layers = list(furnace.get("layers", ()))
        tray_routes = {
            str(tray["tray_id"]): {
                "order_id": tray.get("order_id"),
                "product_unit_id": tray.get("unit_id"),
                "owner": tray.get("owner"),
                "station_id": _OWNER_TO_STATION.get(str(tray.get("owner"))),
                "phase": tray.get("stage"),
                "mold_name": None,
                "comb_name": None,
                "press_locked": False,
            }
            for tray in trays
        }
        workstations = self._workstations(dict(snapshot.get("topology", {})), trays)
        base_snapshot = dict(snapshot)
        all_events = list(base_snapshot.pop("events", ()))
        assignment_events = [
            event
            for event in all_events
            if isinstance(event, Mapping) and event.get("type") == "INSTALL_ASSIGNED"
        ]
        latest_assignment = assignment_events[-1] if assignment_events else None
        scheduler_candidates: list[dict[str, Any]] = []
        scheduler_selected: list[dict[str, Any]] = []
        scheduler_blocked: list[dict[str, Any]] = []
        if latest_assignment is not None:
            task_id = f"{latest_assignment.get('unit_id', '')}:fin_01"
            selected_resource = str(latest_assignment.get("branch", ""))
            explanation = str(latest_assignment.get("explanation_zh", ""))
            for raw_candidate in latest_assignment.get("candidates", ()):
                if not isinstance(raw_candidate, Mapping):
                    continue
                candidate = {
                    "task_id": task_id,
                    "resource_id": str(raw_candidate.get("resource_id", "")),
                    "cost": float(raw_candidate.get("cost", 0.0)),
                    "finish_at": float(raw_candidate.get("finish_at", 0.0)),
                    "queue_wait_s": float(raw_candidate.get("queue_wait_s", 0.0)),
                    "inspection_wait_s": float(raw_candidate.get("inspection_wait_s", 0.0)),
                    "lateness_s": float(raw_candidate.get("lateness_s", 0.0)),
                    "explanation_zh": explanation,
                }
                reason = str(raw_candidate.get("reason", ""))
                if reason:
                    scheduler_blocked.append({**candidate, "reason": reason})
                else:
                    scheduler_candidates.append(candidate)
                if candidate["resource_id"] == selected_resource:
                    scheduler_selected.append(candidate)
        return {
            **base_snapshot,
            "events": all_events[-300:],
            "event_count": len(all_events),
            "line_profile": self.line_profile,
            "status": status,
            "viewer_running": bool(viewer_running),
            "order_id": "" if active is None else str(active.get("order_id", "")),
            "preset": "" if active is None else str(active.get("preset", "")),
            "stage": stage,
            "simulation_speed": float(simulation_speed),
            "simulation_actual_rtf": float(actual_rtf),
            "simulation_speed_saturated": False,
            "arms": arm_states,
            "orders": self._orders(list(snapshot.get("orders", ())), units),
            "workstations": workstations,
            "tray_routes": tray_routes,
            "transfers": {},
            "async_line": {
                "process_router": {"mode": "V2_DUAL_INSTALL"},
                "active_wip": sum(tray.get("owner") != "EMPTY_BUFFER" for tray in trays),
                "wip_limit": len(trays),
                "physical_tray_owners": {str(tray.get("tray_id")): str(tray.get("owner")) for tray in trays},
                "parallelism": {
                    "active_arms": [name for name, item in arm_states.items() if item["status"] == "running"],
                    "current_parallel_arms": sum(item["status"] == "running" for item in arm_states.values()),
                    "max_parallel_arms": 0,
                    "multi_arm_overlap_s": snapshot.get("scheduled_parallel_install_seconds", 0.0),
                },
            },
            "batch": {
                "batch_id": (furnace.get("last_batch") or {}).get("batch_id"),
                "units": [
                    {
                        "layer": layer.get("index"),
                        "phase": "LOCKED" if layer.get("locked") else "EMPTY",
                    }
                    for layer in rack_layers
                ],
            },
            "rack": {
                "shelves": [
                    {
                        "index": layer.get("index"),
                        "state": "LOCKED" if layer.get("locked") else "EMPTY",
                        "lock_engaged": bool(layer.get("locked", False)),
                        "tray_id": layer.get("tray_id"),
                    }
                    for layer in rack_layers
                ]
            },
            "transfer": {
                "phase": "IDLE",
                "step": "",
                "unit_id": None,
                "conveyor_progress": 0.0,
                "lock_position_m": 0.0,
                "moving": False,
            },
            "furnace": {
                **furnace,
                "status": furnace_phase,
                "temperature_c": _FURNACE_TEMPERATURE_C.get(furnace_phase, 25.0),
                "door_open": bool(furnace.get("front_door_open") or furnace.get("rear_door_open")),
            },
            "fixture": {
                "status": "待接入 V2 物理夹具",
                "active_comb_module": None,
                "press_state": "IDLE",
                "clamping_force_n": 0.0,
            },
            "fins": (
                {
                    f"fin_{index:02d}": {
                        "active": True,
                        "inserted": index <= int(active.get("fins_installed", 0)),
                    }
                    for index in range(1, int(active.get("fin_count", 0)) + 1)
                }
                if active is not None
                else {}
            ),
            "paths": (
                {
                    f"path_{index:02d}": {
                        "active": True,
                        "applied": _STAGE_RANK.get(stage, 0) > _STAGE_RANK["DISPENSING"],
                    }
                    for index in range(1, 2 * int(active.get("fin_count", 0)) + 1)
                }
                if active is not None
                else {}
            ),
            "arm2_process": {
                "current_path": (
                    "双喷嘴连续涂覆"
                    if any(operation.get("kind") == "DISPENSING" for operation in operations.values())
                    else ""
                ),
                "completed_paths": (
                    2 * int(active.get("fin_count", 0))
                    if active is not None and _STAGE_RANK.get(stage, 0) > _STAGE_RANK["DISPENSING"]
                    else 0
                ),
                "total_paths": 0 if active is None else 2 * int(active.get("fin_count", 0)),
            },
            "conveyor": {
                "phase": "IDLE",
                "position_m": 0.0,
                "travel_m": 0.0,
                "progress": 0.0,
                "moving": False,
            },
            "inspections": [],
            "kpi": {
                "order_elapsed": float(snapshot.get("sim_time", 0.0)),
                "rework_counts": {},
                "final_quality_score": None,
            },
            "scheduler": {
                "mode": "V2_EARLIEST_FINISH",
                "ready_count": sum(task.get("status") == "READY" for task in tasks),
                "running_count": sum(task.get("status") == "RUNNING" for task in tasks),
                "replan_count": 0,
                "max_assignments_per_tick": 3,
                "selected": scheduler_selected,
                "candidates": scheduler_candidates,
                "blocked_candidates": scheduler_blocked,
            },
            "tasks": tasks,
            "resources_v2": resources_v2,
            "zone_locks": {},
            "faults_v2": [],
            "recoveries": [],
            "manual_fault_requests": [],
            "experiment_metrics": dict(snapshot.get("metrics", {})),
            "motion_plans": [],
            "space_time_reservations": [],
            "motion_blockers": {},
            "gantt_events": [],
            "ui_capabilities": {
                "segments": segment_capabilities,
                "orders": True,
                "custom_orders": False,
                "pause_continue_reset": True,
                "speed": True,
                "fault_injection": False,
            },
        }


__all__ = ["V2StatePresenter"]
