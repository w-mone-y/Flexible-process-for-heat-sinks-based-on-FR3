"""Project the planning DAG onto authoritative physical process feedback.

The manufacturing scheduler is useful for estimating assignments, but its
``TimedSkill`` fallback is not proof that a MuJoCo action has completed.  This
module produces the execution status shown by the UI from the real product,
fixture, transfer and furnace state instead.  A node may become green only
after its corresponding physical completion condition is true.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .planning.task_models import TaskStatus, TaskType, task_status_label_zh

_UNIT_PATTERN = re.compile(r"_UNIT_(\d+)$")

_PRETRANSFER_DONE = {
    "READY_FOR_TRANSFER",
    "TRANSFERRING",
    "LOCKED",
    "BRAZED",
    "UNLOADING",
    "UNLOADED",
    "INSPECTED",
    "DELIVERING",
    "DELIVERED",
}
_RACK_LOCK_DONE = {
    "LOCKED",
    "BRAZED",
    "UNLOADING",
    "UNLOADED",
    "INSPECTED",
    "DELIVERING",
    "DELIVERED",
}
_BRAZED = {"BRAZED", "UNLOADING", "UNLOADED", "INSPECTED", "DELIVERING", "DELIVERED"}
_UNLOADED = {"UNLOADED", "INSPECTED", "DELIVERING", "DELIVERED"}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _unit_index(task: Mapping[str, Any]) -> int | None:
    match = _UNIT_PATTERN.search(str(task.get("unit_id", "")))
    return None if match is None else int(match.group(1)) - 1


def _active_items(items: Any) -> list[Mapping[str, Any]]:
    values = _as_mapping(items).values()
    return [item for item in values if isinstance(item, Mapping) and bool(item.get("active", True))]


def _all_true(items: Iterable[Mapping[str, Any]], field: str) -> bool:
    values = list(items)
    return bool(values) and all(bool(item.get(field, False)) for item in values)


def _legacy_task_types(physical: Mapping[str, Any], active_task_type: str) -> set[str]:
    result = {str(active_task_type)} if active_task_type else set()
    for arm in _as_mapping(physical.get("arms")).values():
        task_type = str(_as_mapping(arm).get("task_type", ""))
        if task_type:
            result.add(task_type)
    return result


class PhysicalTaskStatusProjector:
    """Stateful, monotonic UI projection for one shared MuJoCo workcell."""

    def __init__(self) -> None:
        self._completed: set[str] = set()
        self._cancelled: set[str] = set()

    def reset(self) -> None:
        self._completed.clear()
        self._cancelled.clear()

    @staticmethod
    def _physical_order_id(
        tasks: list[Mapping[str, Any]],
        physical: Mapping[str, Any],
        active_order_id: str | None,
    ) -> str | None:
        if active_order_id:
            return str(active_order_id)
        physical_id = str(physical.get("order_id", ""))
        task_orders = {str(task.get("order_id", "")) for task in tasks if task.get("order_id")}
        if physical_id in task_orders:
            return physical_id
        # Legacy A/B/C entry points independently create their physical and
        # planning IDs.  With one graph in the runtime they still refer to the
        # same physical order, so bind that sole graph explicitly.
        stage = str(physical.get("stage", "IDLE"))
        if len(task_orders) == 1 and stage != "IDLE":
            return next(iter(task_orders))
        return None

    @staticmethod
    def _batch_unit(physical: Mapping[str, Any], index: int | None) -> Mapping[str, Any]:
        if index is None:
            return {}
        units = _as_mapping(physical.get("batch")).get("units", ())
        if not isinstance(units, list) or not 0 <= index < len(units):
            return {}
        return _as_mapping(units[index])

    @staticmethod
    def _is_current_transfer_unit(
        task: Mapping[str, Any], physical: Mapping[str, Any], unit: Mapping[str, Any]
    ) -> bool:
        transfer_unit = str(_as_mapping(physical.get("transfer")).get("unit_id") or "")
        return bool(transfer_unit) and transfer_unit in {
            str(task.get("tray_id") or ""),
            str(unit.get("unit_id") or ""),
        }

    def _assembly_status(
        self,
        task: Mapping[str, Any],
        physical: Mapping[str, Any],
        legacy_types: set[str],
        active_payload: Mapping[str, Any],
        unit_phase: str,
        current_unit: bool,
    ) -> TaskStatus | None:
        task_type = TaskType(str(task.get("task_type")))
        if unit_phase in _PRETRANSFER_DONE:
            return TaskStatus.SUCCEEDED
        if not current_unit:
            return None

        fixture = _as_mapping(physical.get("fixture"))
        fins = _active_items(physical.get("fins"))
        paths = _active_items(physical.get("paths"))
        active_fin_id = str(active_payload.get("fin_id", ""))
        pick_complete = bool(active_payload.get("physical_pick_complete", False))
        place_complete = bool(active_payload.get("physical_place_complete", False))

        if task_type is TaskType.PICK_BASE_PLATE:
            if pick_complete or bool(fixture.get("base_weld_active", False)):
                return TaskStatus.SUCCEEDED
            if "LOAD_BASE" in legacy_types:
                return TaskStatus.RUNNING
        elif task_type is TaskType.PLACE_BASE_PLATE:
            if place_complete or bool(fixture.get("base_weld_active", False)):
                return TaskStatus.SUCCEEDED
            if "LOAD_BASE" in legacy_types and pick_complete:
                return TaskStatus.RUNNING
        elif task_type is TaskType.PREPARE_FIN_TOOL:
            arm1_tool = str(
                _as_mapping(_as_mapping(physical.get("tools")).get("arm1")).get("current_tool") or ""
            )
            if arm1_tool == "parallel_gripper":
                return TaskStatus.SUCCEEDED
            if "PREPARE_FIN_TOOL" in legacy_types:
                return TaskStatus.RUNNING
        elif task_type in {TaskType.DISPENSE_BRAZING, TaskType.REWORK_BRAZING}:
            if _all_true(paths, "applied"):
                return TaskStatus.SUCCEEDED
            expected = "REAPPLY_MATERIAL" if task_type is TaskType.REWORK_BRAZING else "APPLY_MATERIAL"
            if expected in legacy_types or (
                task_type is TaskType.DISPENSE_BRAZING and "REAPPLY_MATERIAL" in legacy_types
            ):
                return TaskStatus.RUNNING
        elif task_type is TaskType.INSPECT_BRAZING:
            if bool(fixture.get("material_passed", False)):
                return TaskStatus.SUCCEEDED
            if "MATERIAL_INSPECT" in legacy_types:
                return TaskStatus.RUNNING
        elif task_type is TaskType.CONFIGURE_COMB:
            if bool(fixture.get("comb_configured", False)) and bool(fixture.get("comb_aligned", False)):
                return TaskStatus.SUCCEEDED
            if "CONFIGURE_COMB" in legacy_types:
                return TaskStatus.RUNNING
        elif task_type in {TaskType.PICK_FIN, TaskType.INSTALL_FIN, TaskType.REINSTALL_FIN}:
            fin_id = str(_as_mapping(task.get("payload")).get("fin_id", ""))
            fin = next((item for item in fins if str(item.get("fin_id")) == fin_id), {})
            if bool(fin.get("inserted", False)):
                return TaskStatus.SUCCEEDED
            is_active_fin = bool({"INSERT_FIN", "ADJUST_FIN"} & legacy_types) and (
                not active_fin_id or active_fin_id == fin_id
            )
            if task_type is TaskType.PICK_FIN:
                if is_active_fin and pick_complete:
                    return TaskStatus.SUCCEEDED
                if is_active_fin:
                    return TaskStatus.RUNNING
            elif task_type in {TaskType.INSTALL_FIN, TaskType.REINSTALL_FIN}:
                if is_active_fin and place_complete:
                    return TaskStatus.SUCCEEDED
                if is_active_fin and pick_complete:
                    return TaskStatus.RUNNING
        elif task_type is TaskType.INSPECT_FINS:
            if bool(fixture.get("fins_passed", False)):
                return TaskStatus.SUCCEEDED
            if "PRE_INSPECT" in legacy_types:
                return TaskStatus.RUNNING
        elif task_type is TaskType.APPLY_PRESS:
            if bool(fixture.get("press_force_held", False)):
                return TaskStatus.SUCCEEDED
            if "PRESS_FIXTURE" in legacy_types:
                return TaskStatus.RUNNING
        elif task_type is TaskType.LOCK_FIXTURE:
            if bool(fixture.get("locked", False)):
                return TaskStatus.SUCCEEDED
            if "LOCK_FIXTURE" in legacy_types:
                return TaskStatus.RUNNING
        return None

    def _logistics_status(
        self,
        task: Mapping[str, Any],
        physical: Mapping[str, Any],
        unit: Mapping[str, Any],
        unit_phase: str,
    ) -> TaskStatus | None:
        task_type = TaskType(str(task.get("task_type")))
        transfer = _as_mapping(physical.get("transfer"))
        step = str(transfer.get("step", ""))
        current_transfer = self._is_current_transfer_unit(task, physical, unit)
        product_stage = str(unit.get("product_stage", ""))

        async_stage_order = {
            "CREATED": 0,
            "BASE_LOADING": 1,
            "MATERIAL_APPLICATION": 2,
            "MATERIAL_INSPECTION": 3,
            "COMB_CONFIGURATION": 4,
            "FIN_ASSEMBLY": 5,
            "PRE_INSPECTION": 6,
            "FIXTURE_PRESSING": 7,
            "FIXTURE_LOCKING": 8,
            "READY_FOR_TRANSFER": 9,
            "FURNACE_LOADING": 10,
            "BRAZING": 11,
            "UNLOADING": 12,
            "POST_INSPECTION": 13,
            "PASS": 14,
            "REWORK_REQUIRED": 14,
            "SCRAPPED": 14,
        }
        stage_index = async_stage_order.get(product_stage, -1)
        if task_type is TaskType.INDEX_EMPTY_TRAY:
            if unit_phase in _PRETRANSFER_DONE or unit_phase == "BUILDING":
                return TaskStatus.SUCCEEDED
        elif task_type is TaskType.TRANSFER_S1_S2A:
            if unit_phase in _PRETRANSFER_DONE or stage_index >= 2:
                return TaskStatus.SUCCEEDED
            if stage_index >= 1:
                return TaskStatus.RUNNING
        elif task_type is TaskType.TRANSFER_S2A_S2B:
            if unit_phase in _PRETRANSFER_DONE or stage_index >= 3:
                return TaskStatus.SUCCEEDED
            if stage_index >= 2:
                return TaskStatus.RUNNING
        elif task_type is TaskType.TRANSFER_S2B_S3:
            if unit_phase in _PRETRANSFER_DONE or stage_index >= 4:
                return TaskStatus.SUCCEEDED
            if stage_index >= 3:
                return TaskStatus.RUNNING
        elif task_type is TaskType.TRANSFER_S3_RACK:
            if unit_phase in _RACK_LOCK_DONE:
                return TaskStatus.SUCCEEDED
            if unit_phase in {"READY_FOR_TRANSFER", "TRANSFERRING"}:
                return TaskStatus.RUNNING

        if task_type in {
            TaskType.TRANSFER_TRAY_OUT,
            TaskType.MOVE_ELEVATOR,
            TaskType.LOAD_RACK_LAYER,
            TaskType.LOCK_RACK_LAYER,
        }:
            if unit_phase in _RACK_LOCK_DONE:
                return TaskStatus.SUCCEEDED
            if not current_transfer or not step.startswith("load_"):
                return None
            if task_type is TaskType.TRANSFER_TRAY_OUT:
                return TaskStatus.RUNNING if step == "load_outfeed" else TaskStatus.SUCCEEDED
            if task_type is TaskType.MOVE_ELEVATOR:
                if step in {"load_outfeed"}:
                    return None
                return TaskStatus.RUNNING if step in {"load_lift", "load_align"} else TaskStatus.SUCCEEDED
            if task_type is TaskType.LOAD_RACK_LAYER:
                if step in {"load_push"}:
                    return TaskStatus.RUNNING
                if step in {"load_lock", "load_retract", "load_parallel_return"}:
                    return TaskStatus.SUCCEEDED
                return None
            if task_type is TaskType.LOCK_RACK_LAYER:
                if step == "load_lock":
                    return TaskStatus.RUNNING
                if step in {"load_retract", "load_parallel_return"}:
                    return TaskStatus.SUCCEEDED
                return None

        if task_type is TaskType.UNLOAD_RACK_LAYER:
            if unit_phase in _UNLOADED:
                return TaskStatus.SUCCEEDED
            if current_transfer and step.startswith("unload_"):
                return TaskStatus.RUNNING
        elif task_type is TaskType.POST_BRAZE_INSPECTION:
            if unit_phase in {"INSPECTED", "DELIVERING", "DELIVERED"}:
                return TaskStatus.SUCCEEDED
            arms = _as_mapping(physical.get("arms"))
            arm3_type = str(_as_mapping(arms.get("arm3")).get("task_type", ""))
            if unit_phase == "UNLOADED" and arm3_type == "POST_INSPECT":
                return TaskStatus.RUNNING
        return None

    @staticmethod
    def _batch_status(task: Mapping[str, Any], physical: Mapping[str, Any]) -> TaskStatus | None:
        task_type = TaskType(str(task.get("task_type")))
        batch = _as_mapping(physical.get("batch"))
        batch_stage = str(batch.get("stage", physical.get("stage", "")))
        units = batch.get("units", ())
        unit_phases = (
            [str(_as_mapping(item).get("phase", "")) for item in units] if isinstance(units, list) else []
        )

        if task_type is TaskType.BATCH_READY:
            if batch_stage in {"READY_FOR_BRAZING", "BRAZING", "UNLOADING", "POST_INSPECTION", "COMPLETE"}:
                return TaskStatus.SUCCEEDED
            if unit_phases and all(phase in _RACK_LOCK_DONE for phase in unit_phases):
                return TaskStatus.RUNNING
        elif task_type is TaskType.RUN_FURNACE:
            if batch_stage in {"UNLOADING", "POST_INSPECTION", "COMPLETE"} or (
                unit_phases and all(phase in _BRAZED for phase in unit_phases)
            ):
                return TaskStatus.SUCCEEDED
            if batch_stage in {"READY_FOR_BRAZING", "BRAZING"}:
                return TaskStatus.RUNNING
        return None

    @staticmethod
    def _route_status(task: Mapping[str, Any], unit: Mapping[str, Any]) -> TaskStatus | None:
        task_type = TaskType(str(task.get("task_type")))
        if task_type not in {TaskType.ROUTE_PASS, TaskType.ROUTE_REWORK, TaskType.ROUTE_SCRAP}:
            return None
        phase = str(unit.get("phase", ""))
        if phase not in {"INSPECTED", "DELIVERING", "DELIVERED"}:
            return None
        disposition = str(unit.get("disposition") or "")
        condition = str(_as_mapping(task.get("payload")).get("condition", ""))
        if condition != disposition:
            return TaskStatus.CANCELLED
        # The selected route is the physical finished-goods delivery task.
        # Keep it orange while the gate is open and tray is entering; only the
        # closed-gate handoff is authoritative completion evidence.
        return TaskStatus.SUCCEEDED if phase == "DELIVERED" else TaskStatus.RUNNING

    def _evidence_status(
        self,
        task: Mapping[str, Any],
        physical: Mapping[str, Any],
        legacy_types: set[str],
        active_payload: Mapping[str, Any],
    ) -> TaskStatus | None:
        task_type = TaskType(str(task.get("task_type")))
        if task_type in {TaskType.BATCH_READY, TaskType.RUN_FURNACE} and _as_mapping(physical.get("batch")):
            return self._batch_status(task, physical)

        index = _unit_index(task)
        unit = self._batch_unit(physical, index)
        unit_phase = str(unit.get("phase", ""))
        batch = _as_mapping(physical.get("batch"))
        active_unit = next(
            (
                position
                for position, item in enumerate(batch.get("units", ()))
                if isinstance(item, Mapping)
                and str(item.get("phase", "")) in {"BUILDING", "READY_FOR_TRANSFER", "TRANSFERRING"}
            ),
            None,
        )
        current_unit = not batch or index is None or index == active_unit

        route = self._route_status(task, unit)
        if route is not None:
            return route
        logistics = self._logistics_status(task, physical, unit, unit_phase)
        if logistics is not None:
            return logistics
        assembly_types = {
            TaskType.PICK_BASE_PLATE,
            TaskType.PLACE_BASE_PLATE,
            TaskType.PREPARE_FIN_TOOL,
            TaskType.CONFIGURE_COMB,
            TaskType.DISPENSE_BRAZING,
            TaskType.INSPECT_BRAZING,
            TaskType.REWORK_BRAZING,
            TaskType.PICK_FIN,
            TaskType.INSTALL_FIN,
            TaskType.REINSTALL_FIN,
            TaskType.INSPECT_FINS,
            TaskType.APPLY_PRESS,
            TaskType.LOCK_FIXTURE,
        }
        if task_type in assembly_types:
            return self._assembly_status(
                task,
                physical,
                legacy_types,
                active_payload,
                unit_phase,
                current_unit,
            )

        # Legacy single-order mode uses a conveyor instead of the rack actor.
        # These conservative stage gates still prevent premature completion.
        if not batch:
            stage = str(physical.get("stage", ""))
            if task_type in {
                TaskType.TRANSFER_TRAY_OUT,
                TaskType.MOVE_ELEVATOR,
                TaskType.LOAD_RACK_LAYER,
                TaskType.LOCK_RACK_LAYER,
                TaskType.BATCH_READY,
            }:
                if stage in {
                    "BRAZING",
                    "UNLOADING",
                    "POST_INSPECTION",
                    "PASS",
                    "REWORK_REQUIRED",
                    "SCRAPPED",
                }:
                    return TaskStatus.SUCCEEDED
                if stage == "FURNACE_LOADING":
                    return TaskStatus.RUNNING
            elif task_type is TaskType.RUN_FURNACE:
                if stage in {"UNLOADING", "POST_INSPECTION", "PASS", "REWORK_REQUIRED", "SCRAPPED"}:
                    return TaskStatus.SUCCEEDED
                if stage == "BRAZING":
                    return TaskStatus.RUNNING
            elif task_type is TaskType.UNLOAD_RACK_LAYER:
                if stage in {"POST_INSPECTION", "PASS", "REWORK_REQUIRED", "SCRAPPED"}:
                    return TaskStatus.SUCCEEDED
                if stage == "UNLOADING":
                    return TaskStatus.RUNNING
            elif task_type is TaskType.POST_BRAZE_INSPECTION:
                if stage in {"PASS", "REWORK_REQUIRED", "SCRAPPED"}:
                    return TaskStatus.SUCCEEDED
                if stage == "POST_INSPECTION":
                    return TaskStatus.RUNNING
        return None

    def project(
        self,
        tasks: list[Mapping[str, Any]],
        physical: Mapping[str, Any],
        *,
        active_task_type: str = "",
        active_task_payload: Mapping[str, Any] | None = None,
        active_order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return UI task dictionaries whose colors follow physical truth."""

        # Only top-level presentation fields are overwritten below. Payload,
        # predecessor and successor containers are read-only, so a shallow
        # copy avoids recursively cloning the entire DAG every UI refresh.
        source = [dict(task) for task in tasks]
        bound_order = self._physical_order_id(source, physical, active_order_id)
        legacy_types = _legacy_task_types(physical, active_task_type)
        payload = _as_mapping(active_task_payload)
        projected_status: dict[str, TaskStatus] = {}

        for task in source:
            task_id = str(task.get("task_id", ""))
            task["scheduler_status"] = str(task.get("status", TaskStatus.PENDING.value))
            if task_id in self._completed:
                status = TaskStatus.SUCCEEDED
            elif task_id in self._cancelled:
                status = TaskStatus.CANCELLED
            elif bound_order is None or str(task.get("order_id", "")) != bound_order:
                status = TaskStatus.PENDING
            else:
                status = self._evidence_status(task, physical, legacy_types, payload) or TaskStatus.PENDING
            projected_status[task_id] = status

        # READY is derived from the physically projected predecessor states,
        # never from the scheduler's independent duration estimates.
        for task in source:
            task_id = str(task.get("task_id", ""))
            status = projected_status[task_id]
            if status is not TaskStatus.PENDING:
                continue
            predecessors = [str(value) for value in task.get("predecessors", ())]
            if predecessors and all(
                projected_status.get(item) is TaskStatus.SUCCEEDED for item in predecessors
            ):
                projected_status[task_id] = TaskStatus.READY
            elif not predecessors and bound_order == str(task.get("order_id", "")):
                projected_status[task_id] = TaskStatus.READY

        result: list[dict[str, Any]] = []
        for task in source:
            task_id = str(task.get("task_id", ""))
            status = projected_status[task_id]
            if status is TaskStatus.SUCCEEDED:
                self._completed.add(task_id)
            elif status is TaskStatus.CANCELLED:
                self._cancelled.add(task_id)
            task["status"] = status.value
            task["status_zh"] = task_status_label_zh(status)
            task["status_source"] = "PHYSICAL"
            result.append(task)
        return result


__all__ = ["PhysicalTaskStatusProjector"]
