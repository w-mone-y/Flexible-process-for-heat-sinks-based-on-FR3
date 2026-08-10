"""ManufacturingRuntime authority with the existing V2 cell as its actor adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..execution import PhysicalCompletionEvidence, SkillExecutionResult, SkillRegistry
from ..flexible import ProcessPlan, build_inline_plan
from ..manufacturing_runtime import ManufacturingRuntime
from ..planning import ManufacturingTask, TaskType, V2_DUAL_INSTALL_PROFILE
from ..scheduling import ResourceStatus
from .furnace import FurnacePhase
from .runtime import DualLineRuntime, RuntimeExecutionGate, UnitStage
from .tray_flow import TrayOwner

_STAGE_RANK = {stage: index for index, stage in enumerate(UnitStage)}


_OPERATION_PERMITS: dict[TaskType, tuple[str, ...]] = {
    TaskType.PICK_BASE_PLATE: ("BASE_LOADING",),
    TaskType.DISPENSE_BRAZING: ("DISPENSING",),
    TaskType.INSPECT_BRAZING: ("MATERIAL_INSPECTION",),
    TaskType.REVIEW_BRAZING_CLOSEUP: ("MATERIAL_INSPECTION",),
    TaskType.PICK_FIN: ("INSTALL_FIN",),
    TaskType.INSPECT_FINS: ("PRE_BRAZE_INSPECTION",),
    TaskType.REVIEW_FINS_CLOSEUP: ("PRE_BRAZE_INSPECTION",),
    TaskType.MOVE_ELEVATOR: ("FURNACE_FRONT_OPEN",),
    TaskType.LOAD_RACK_LAYER: ("FURNACE_LOAD_TRAY",),
    TaskType.RUN_FURNACE: ("FURNACE_FRONT_CLOSE",),
    TaskType.UNLOAD_RACK_LAYER: (
        "FURNACE_REAR_OPEN",
        "FURNACE_UNLOAD_TRAY",
        "FURNACE_REAR_CLOSE",
    ),
    TaskType.POST_BRAZE_INSPECTION: ("POST_BRAZE_INSPECTION",),
    TaskType.ROUTE_PASS: (
        "OUTPUT_GATE_OPEN",
        "OUTPUT_DELIVERY",
        "OUTPUT_GATE_CLOSE",
        "VIRTUAL_RETURN",
    ),
    TaskType.ROUTE_REWORK: ("OUTPUT_GATE_OPEN", "OUTPUT_DELIVERY", "OUTPUT_GATE_CLOSE"),
    TaskType.ROUTE_SCRAP: ("OUTPUT_GATE_OPEN", "OUTPUT_DELIVERY", "OUTPUT_GATE_CLOSE"),
}


def _fin_index(task: ManufacturingTask) -> int:
    value = str(task.payload.get("fin_id", ""))
    digits = "".join(character for character in value if character.isdigit())
    return int(digits or 0)


@dataclass(slots=True)
class _V2TaskSkill:
    bridge: "V2PhysicalExecutionBridge"
    task: ManufacturingTask | None = None
    resource_id: str = ""
    started_at: float = 0.0

    def start(self, task: ManufacturingTask, resource_id: str, context: Any, now: float) -> None:
        del context
        self.task = task
        self.resource_id = str(resource_id).upper()
        self.started_at = float(now)
        self.bridge.authorize(task, self.resource_id)

    @staticmethod
    def execution_timeout(task: ManufacturingTask) -> float:
        # A globally scheduled task may legitimately wait for WIP admission,
        # a shared merge corridor or a partial furnace batch. This is a safety
        # timeout only; measured completion remains mandatory.
        return max(600.0, float(task.estimated_duration) * 20.0)

    def update(self, now: float, dt: float) -> SkillExecutionResult:
        del dt
        if self.task is None:
            return SkillExecutionResult.success()
        complete, checks = self.bridge.task_complete(
            self.task,
            self.resource_id,
            self.started_at,
        )
        if not complete:
            # Estimated duration may animate progress and keep the watchdog
            # informed, but it is deliberately capped below 100%; only the
            # measured predicate below can complete the task.
            elapsed = max(0.0, float(now) - self.started_at)
            estimate = max(0.1, float(self.task.estimated_duration))
            return SkillExecutionResult.running_result({"progress": 0.95 * elapsed / (elapsed + estimate)})
        evidence = self.bridge.completion_evidence(float(now), checks)
        return SkillExecutionResult.success(
            {"physical_checks": list(checks)},
            completion_evidence=evidence,
        )

    def cancel(self, task_id: str) -> None:
        if self.task is not None and self.task.task_id == task_id:
            self.bridge.revoke(task_id)
            self.task = None


class V2PhysicalExecutionBridge(RuntimeExecutionGate):
    """Translate scheduled DAG tasks into permits and measured V2 milestones."""

    def __init__(self, runtime: DualLineRuntime) -> None:
        self.runtime = runtime
        self.physical_gate: RuntimeExecutionGate | None = None
        self.manufacturing_runtime: ManufacturingRuntime | None = None
        self._permits: dict[tuple[str, str], dict[str, str]] = {}
        self._task_permits: dict[str, set[tuple[str, str]]] = {}

    def bind_physical_gate(self, gate: RuntimeExecutionGate | None) -> None:
        if gate is self:
            return
        self.physical_gate = gate

    def build_registry(self) -> SkillRegistry:
        registry = SkillRegistry()
        for task_type in TaskType:
            registry.register_factory(
                task_type,
                lambda bridge=self: _V2TaskSkill(bridge),
                requires_physical_evidence=True,
            )
        return registry

    def authorize(self, task: ManufacturingTask, resource_id: str) -> None:
        keys: set[tuple[str, str]] = set()
        for kind in _OPERATION_PERMITS.get(task.task_type, ()):
            key = (task.unit_id, kind)
            self._permits.setdefault(key, {})[task.task_id] = resource_id
            keys.add(key)
        self._task_permits[task.task_id] = keys

    def revoke(self, task_id: str) -> None:
        for key in self._task_permits.pop(task_id, set()):
            holders = self._permits.get(key)
            if holders is None:
                continue
            holders.pop(task_id, None)
            if not holders:
                self._permits.pop(key, None)

    def tray_ready(self, tray_id: str, owner: TrayOwner) -> bool:
        if self.physical_gate is None:
            return True
        return self.physical_gate.tray_ready(tray_id, owner)

    def owner_available(self, owner: TrayOwner) -> bool:
        if self.physical_gate is None:
            return True
        return self.physical_gate.owner_available(owner)

    def operation_complete(self, resource: str, unit_id: str, kind: str) -> bool:
        if self.physical_gate is None:
            return True
        return self.physical_gate.operation_complete(resource, unit_id, kind)

    def operation_start_allowed(self, resource: str, unit_id: str, kind: str) -> bool:
        holders = self._permits.get((unit_id, kind), {})
        # MERGING is a first-class ownership-controlled transport, not Arm3's
        # inspection task. Tying it to the camera resource creates a circular
        # wait when another pallet already occupies the single merge corridor.
        permitted = kind in {
            "MERGING",
            "OUTPUT_GATE_OPEN",
            "OUTPUT_DELIVERY",
            "OUTPUT_GATE_CLOSE",
            "VIRTUAL_RETURN",
        } or bool(holders)
        if not permitted and kind in {"MATERIAL_INSPECTION", "PRE_BRAZE_INSPECTION"}:
            permitted = any(
                key_kind in {"MATERIAL_INSPECTION", "PRE_BRAZE_INSPECTION"} and values
                for (_unit, key_kind), values in self._permits.items()
            )
        if not permitted and kind in {"FURNACE_LOAD_TRAY", "FURNACE_UNLOAD_TRAY"}:
            permitted = any(
                key_kind == kind and values for (_unit, key_kind), values in self._permits.items()
            )
        if (
            holders
            and permitted
            and kind
            in {
                "BASE_LOADING",
                "DISPENSING",
                "MATERIAL_INSPECTION",
                "PRE_BRAZE_INSPECTION",
            }
        ):
            permitted = str(resource).upper() in set(holders.values())
        if not permitted and kind in {
            "FURNACE_FRONT_OPEN",
            "FURNACE_FRONT_CLOSE",
            "FURNACE_REAR_OPEN",
            "FURNACE_REAR_CLOSE",
        }:
            # Door motion is a batch-wide synchronization operation. The
            # leader selected by the physical furnace need not belong to the
            # same order whose batch task obtained the shared-door resource.
            permitted = any(
                key_kind == kind and holders for (_unit, key_kind), holders in self._permits.items()
            )
        if not permitted:
            return False
        callback = (
            None
            if self.physical_gate is None
            else getattr(self.physical_gate, "operation_start_allowed", None)
        )
        return True if callback is None else bool(callback(resource, unit_id, kind))

    def unit_admission_allowed(self, unit_id: str) -> bool:
        """Allow physical S1 admission only for a unit released by the DAG.

        The physical V2 runtime still owns motion and tray transfer, but it
        must not independently choose a higher-priority queued unit than the
        manufacturing scheduler selected.  Without this gate a burst such as
        A/B/C/D could admit D physically while the DAG was running C, leaving
        both authorities waiting on different trays.
        """

        if self.manufacturing_runtime is None:
            # During construction the bridge is used by the physical runtime
            # before the manufacturing facade is attached; keep legacy
            # standalone DualLineRuntime behaviour unchanged.
            return True
        return any(unit_id in entry.admitted_unit_ids for entry in self.manufacturing_runtime.orders.values())

    def preferred_install_resource(self, unit_id: str) -> str | None:
        """Return the resource selected by the global DAG scheduler's OR branch."""

        resources = sorted(set(self._permits.get((str(unit_id), "INSTALL_FIN"), {}).values()))
        return next((item for item in resources if item in {"ARM1", "ARM3"}), None)

    def _event(
        self,
        event_type: str,
        unit_id: str,
        *,
        since: float,
        kind: str | None = None,
        fin_index: int | None = None,
    ) -> bool:
        return any(
            event.get("type") == event_type
            and event.get("unit_id") == unit_id
            and float(event.get("time", -1.0)) + 1.0e-9 >= since
            and (kind is None or event.get("kind") == kind)
            and (fin_index is None or int(event.get("fin_index", -1)) == fin_index)
            for event in self.runtime.events
        )

    def _milestone(self, resource: str, unit_id: str, kind: str, milestone: str) -> bool:
        callback = (
            None if self.physical_gate is None else getattr(self.physical_gate, "operation_milestone", None)
        )
        return False if callback is None else bool(callback(resource, unit_id, kind, milestone))

    def task_complete(
        self,
        task: ManufacturingTask,
        resource_id: str,
        since: float,
    ) -> tuple[bool, tuple[str, ...]]:
        unit = self.runtime.units.get(task.unit_id)
        if unit is None:
            return False, ()
        kind = task.task_type

        def event(name: str) -> bool:
            return self._event(
                "OPERATION_COMPLETED",
                task.unit_id,
                since=since,
                kind=name,
            )

        if kind is TaskType.PICK_BASE_PLATE:
            done = self._milestone("ARM1", task.unit_id, "BASE_LOADING", "grasp") or event("BASE_LOADING")
            return done, ("吸盘负压建立", "基板已离开取料位")
        if kind is TaskType.PLACE_BASE_PLATE:
            # PICK/PLACE are two DAG milestones inside one non-preemptible
            # physical operation. The release event can be emitted in the
            # same tick that PICK turns terminal, just before PLACE starts.
            done = self._event(
                "OPERATION_COMPLETED",
                task.unit_id,
                since=0.0,
                kind="BASE_LOADING",
            )
            return done, ("基板已释放", "基板在托盘上停稳")
        if kind is TaskType.DISPENSE_BRAZING:
            return event("DISPENSING"), ("全部规划钎料轨迹完成",)
        if kind is TaskType.INSPECT_BRAZING:
            done = self._event(
                "OPERATION_COMPLETED",
                task.unit_id,
                since=0.0,
                kind="MATERIAL_INSPECTION",
            )
            return done, ("S2B图像分析完成",)
        if kind is TaskType.REVIEW_BRAZING_CLOSEUP:
            done = self.runtime.camera_coordination.route_review_completed(
                task.unit_id, "MATERIAL_INSPECTION"
            )
            return done, ("S3B钎料近景复核完成",)
        if kind is TaskType.PICK_FIN:
            done = self._milestone(resource_id, task.unit_id, "INSTALL_FIN", "grasp") or self._event(
                "FIN_INSTALLED",
                task.unit_id,
                since=since,
                fin_index=_fin_index(task),
            )
            return done, ("夹爪间距匹配翅片厚度", "翅片已离开料台")
        if kind is TaskType.INSTALL_FIN:
            done = self._event(
                "FIN_INSTALLED",
                task.unit_id,
                since=0.0,
                fin_index=_fin_index(task),
            )
            return done, ("目标翅片已释放", "翅片在梳齿槽中停稳")
        if kind is TaskType.INSPECT_FINS:
            done = self._event(
                "OPERATION_COMPLETED",
                task.unit_id,
                since=0.0,
                kind="PRE_BRAZE_INSPECTION",
            )
            return done, ("S4焊前图像分析完成",)
        if kind is TaskType.REVIEW_FINS_CLOSEUP:
            done = self.runtime.camera_coordination.route_review_completed(
                task.unit_id, "PRE_BRAZE_INSPECTION"
            )
            return done, ("S3B翅片近景复核完成",)
        if kind is TaskType.MOVE_ELEVATOR:
            done = bool(self.runtime.furnace.state.front_door_open) or self._event(
                "FURNACE_FRONT_DOOR_OPENED",
                task.unit_id,
                since=0.0,
            )
            return done, (
                "炉前门完全打开",
                "装载机构对位",
            )
        if kind is TaskType.LOAD_RACK_LAYER:
            done = _STAGE_RANK[unit.stage] >= _STAGE_RANK[UnitStage.BRAZING]
            return done, ("托盘已进入目标炉层",)
        if kind is TaskType.LOCK_RACK_LAYER:
            return _STAGE_RANK[unit.stage] >= _STAGE_RANK[UnitStage.BRAZING], ("炉层锁定",)
        if kind is TaskType.RUN_FURNACE:
            done = self.runtime.furnace.state.phase in {
                FurnacePhase.READY_TO_UNLOAD,
                FurnacePhase.UNLOADING,
                FurnacePhase.COMPLETE,
            }
            return done, ("热循环结束", "炉温进入可卸载区间")
        if kind is TaskType.UNLOAD_RACK_LAYER:
            done = _STAGE_RANK[unit.stage] >= _STAGE_RANK[UnitStage.POST_BRAZE_INSPECTION]
            return done, ("托盘从后门完全卸出",)
        if kind is TaskType.POST_BRAZE_INSPECTION:
            return event("POST_BRAZE_INSPECTION"), ("焊后固定相机分析完成",)
        if kind in {TaskType.ROUTE_PASS, TaskType.ROUTE_REWORK, TaskType.ROUTE_SCRAP}:
            done = self._event("UNIT_COMPLETED", task.unit_id, since=0.0)
            return done, ("成品进入出口箱", "空托盘所有权已回收")
        if kind is TaskType.CONFIGURE_COMB:
            return _STAGE_RANK[unit.stage] >= _STAGE_RANK[UnitStage.FIN_INSTALLATION], (
                "订单对应梳齿配置已切换",
            )
        if kind in {TaskType.APPLY_PRESS, TaskType.LOCK_FIXTURE}:
            return _STAGE_RANK[unit.stage] >= _STAGE_RANK[UnitStage.WAITING_BUFFER], ("实体工装状态已确认",)
        if kind in {TaskType.TRANSFER_TRAY_OUT, TaskType.BATCH_READY}:
            return _STAGE_RANK[unit.stage] >= _STAGE_RANK[UnitStage.FURNACE_BUFFER], (
                "载件托盘已到炉前缓存并停稳",
            )
        # Pure control milestones (tool/capability reservation and conditional
        # routing nodes) do not command geometry. They still complete through
        # this common port, but are explicitly identified as control evidence.
        return True, ("统一运行时控制谓词已满足",)

    def completion_evidence(
        self,
        observed_at: float,
        checks: tuple[str, ...],
    ) -> PhysicalCompletionEvidence | None:
        if self.physical_gate is None:
            return None
        return PhysicalCompletionEvidence(
            observed_at=observed_at,
            source="mujoco:v2_execution_gate",
            checks=checks or ("物理反馈谓词已满足",),
        )

    def task_dispatch_allowed(self, task: ManufacturingTask) -> tuple[bool, str]:
        """Apply physical WIP/batch feasibility before resource reservation."""

        if task.task_type is TaskType.LOAD_RACK_LAYER:
            position = self.runtime._furnace_load_position
            queue = self.runtime._furnace_load_queue
            if position < len(queue):
                allowed = queue[position] == task.unit_id
                return allowed, ("" if allowed else "等待前方托盘按物理缓存顺序入炉")
        if task.task_type is TaskType.UNLOAD_RACK_LAYER:
            occupied = [layer for layer in self.runtime.furnace.state.layers if layer.tray_id is not None]
            if occupied:
                next_tray = max(occupied, key=lambda layer: layer.index).tray_id
                next_unit = next(
                    (unit.unit_id for unit in self.runtime.units.values() if unit.tray_id == next_tray),
                    None,
                )
                allowed = next_unit == task.unit_id
                if not allowed:
                    return False, "等待更高物理炉层先完成卸载"
                # A closed rear door is not a blocker: the physical actor
                # needs this logical permit to open it first.  Once the door
                # is open, the same task continues to the unload transfer.
                if "FURNACE_TRANSFER" in self.runtime.operations:
                    return False, "等待上一托盘完成炉后移载"
                if "POST_CAMERA" in self.runtime.operations:
                    return False, "等待焊后检测位释放"
                if not self.runtime._owner_free(TrayOwner.POST_SCAN):
                    return False, "等待焊后检测托盘离开"
                if not self.runtime._target_available(TrayOwner.POST_SCAN):
                    return False, "等待焊后检测工位可接收托盘"
                return True, ""
        if task.task_type is TaskType.INSPECT_FINS:
            unit = self.runtime.units.get(task.unit_id)
            allowed = bool(
                unit is not None and _STAGE_RANK[unit.stage] >= _STAGE_RANK[UnitStage.PRE_BRAZE_INSPECTION]
            )
            return allowed, ("" if allowed else "等待托盘完成合流并到达S4")
        if task.task_type is TaskType.REVIEW_BRAZING_CLOSEUP:
            unit = self.runtime.units.get(task.unit_id)
            allowed = bool(
                unit is not None
                and (
                    unit.stage is UnitStage.BRAZING_REVIEW
                    or self.runtime.camera_coordination.route_review_completed(
                        task.unit_id, "MATERIAL_INSPECTION"
                    )
                )
            )
            return allowed, ("" if allowed else "等待托盘到达S3B并完成基板近景复核")
        if task.task_type is TaskType.REVIEW_FINS_CLOSEUP:
            unit = self.runtime.units.get(task.unit_id)
            allowed = bool(
                unit is not None
                and (
                    unit.stage is UnitStage.FINS_REVIEW
                    or self.runtime.camera_coordination.route_review_completed(
                        task.unit_id, "PRE_BRAZE_INSPECTION"
                    )
                )
            )
            return allowed, ("" if allowed else "等待翅片安装完成并到达S3B")
        if task.task_type is not TaskType.RUN_FURNACE:
            return True, ""
        batch_ready_stages = {
            UnitStage.BRAZING,
            UnitStage.FURNACE_UNLOADING,
            UnitStage.POST_BRAZE_INSPECTION,
            UnitStage.WAITING_OUTPUT,
            UnitStage.DELIVERING,
            UnitStage.PRODUCT_REMOVED,
            UnitStage.VIRTUAL_RETURN,
            UnitStage.COMPLETE,
        }
        active_batch = tuple(self.runtime._active_batch_units)
        if not active_batch or task.unit_id not in active_batch:
            return False, "等待所属物理炉批完成装载"
        waiting = [
            unit_id for unit_id in active_batch if self.runtime.units[unit_id].stage not in batch_ready_stages
        ]
        if waiting:
            return False, "等待同一滚动炉批的上游托盘到达炉前缓存"
        return True, ""


class UnifiedV2Runtime:
    """Compatibility facade: one manufacturing authority, one V2 physical adapter."""

    def __init__(self, *, fast: bool = False) -> None:
        self.physical_runtime = DualLineRuntime(fast=fast)
        self.bridge = V2PhysicalExecutionBridge(self.physical_runtime)
        self.physical_runtime.set_execution_gate(self.bridge)
        self.manufacturing_runtime = ManufacturingRuntime(
            scheduler_mode="dynamic",
            skill_registry=self.bridge.build_registry(),
            context=self.bridge,
            max_wip_units=6,
            camera_coordination=True,
            enable_motion_planning=True,
            execution_profile=V2_DUAL_INSTALL_PROFILE,
            dispatch_guard=self.bridge.task_dispatch_allowed,
            external_batch_controller=True,
        )
        self.bridge.manufacturing_runtime = self.manufacturing_runtime
        # Arm3's V2 head is a fixed camera + narrow-gripper composite, so fin
        # work does not pay the removable-tool change cost used by V1.
        # ``parallel_gripper`` is the task-level GRIPPER class token retained
        # by the legacy graph builder; physically it resolves to Arm3's fixed
        # narrow-gripper half of the hybrid head.
        self.manufacturing_runtime.resources.get("ARM3").current_tool = "parallel_gripper"

    def _sync_physical_resources(self) -> None:
        isolated = {str(resource).upper() for resource in self.physical_runtime.faults.isolated_resources}
        for resource_id, state in self.manufacturing_runtime.resources.states.items():
            if resource_id in isolated:
                if state.current_task_id is None:
                    state.status = ResourceStatus.OFFLINE
                    state.fault_code = "ARM_UNAVAILABLE"
                continue
            if state.status is ResourceStatus.OFFLINE:
                state.status = ResourceStatus.IDLE
                state.fault_code = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.physical_runtime, name)

    def set_execution_gate(self, gate: RuntimeExecutionGate | None) -> None:
        self.bridge.bind_physical_gate(gate)
        self.physical_runtime.set_execution_gate(self.bridge)

    def submit_order(
        self,
        preset: str,
        *,
        order_id: str | None = None,
        quantity: int = 1,
        priority: int = 10,
        due_at: float | None = None,
        urgent: bool = False,
        route_strategy: str = "STANDARD",
    ):
        identifier = str(order_id or "").strip() or self.next_order_id
        plan = build_inline_plan(
            preset=str(preset).strip().upper(),
            order_id=identifier,
            quantity=int(quantity),
            priority=int(priority),
            route_strategy=str(route_strategy).strip().upper() or "STANDARD",
        )
        return self.submit_plan(plan, due_at=due_at, urgent=urgent)

    def submit_plan(self, plan: ProcessPlan, *, due_at: float | None = None, urgent: bool = False):
        # Defer physical admission until the next common tick. This lets a
        # burst of normal/urgent orders enter the global scheduler before S1 is
        # occupied; otherwise the first submitted order could block a higher
        # priority order that ManufacturingRuntime selected in the same UI
        # frame.
        order = self.physical_runtime.submit_plan(
            plan,
            due_at=due_at,
            urgent=urgent,
            dispatch=False,
        )
        self._sync_physical_resources()
        entry = self.manufacturing_runtime.submit_plan(plan, urgent=urgent, now=self.sim_time)
        self._apply_v2_station_zones(entry.graph_task_ids)
        return order

    def _apply_v2_station_zones(self, task_ids: tuple[str, ...]) -> None:
        """Replace V1's monolithic Table2 lock with V2 station semantics."""

        zones = {
            TaskType.PICK_BASE_PLATE: ("ZONE_S1_ARM1", "ZONE_BASE_MAGAZINE"),
            TaskType.PLACE_BASE_PLATE: ("ZONE_S1_ARM1",),
            TaskType.DISPENSE_BRAZING: ("ZONE_S2A_ARM2",),
            TaskType.INSPECT_BRAZING: ("ZONE_S2B_ARM3",),
            TaskType.REVIEW_BRAZING_CLOSEUP: ("ZONE_S3B_ARM3",),
            TaskType.CONFIGURE_COMB: ("ZONE_S3_SHARED",),
            # V2 has two physically separate fin magazines and installation
            # tables. Their collision safety is enforced by the selected
            # branch actor and merge gate, not V1's monolithic S3 lock.
            TaskType.PICK_FIN: (),
            TaskType.INSTALL_FIN: (),
            TaskType.INSPECT_FINS: ("ZONE_S4_SHARED",),
            TaskType.REVIEW_FINS_CLOSEUP: ("ZONE_S3B_ARM3",),
            TaskType.APPLY_PRESS: ("ZONE_S4_SHARED",),
            TaskType.LOCK_FIXTURE: ("ZONE_S4_SHARED",),
        }
        stations = {
            TaskType.PICK_BASE_PLATE: "S1_BASE_LOADING",
            TaskType.PLACE_BASE_PLATE: "S1_BASE_LOADING",
            TaskType.DISPENSE_BRAZING: "S2A_DISPENSING",
            TaskType.INSPECT_BRAZING: "S2B_MATERIAL_INSPECTION",
            TaskType.REVIEW_BRAZING_CLOSEUP: "S3B_ARM3_INSTALL",
            TaskType.CONFIGURE_COMB: "S3_DUAL_INSTALL",
            TaskType.PICK_FIN: "S3_DUAL_INSTALL",
            TaskType.INSTALL_FIN: "S3_DUAL_INSTALL",
            TaskType.INSPECT_FINS: "S4_PRE_BRAZE_INSPECTION",
            TaskType.REVIEW_FINS_CLOSEUP: "S3B_ARM3_INSTALL",
            TaskType.APPLY_PRESS: "S4_PRE_BRAZE_INSPECTION",
            TaskType.LOCK_FIXTURE: "S4_PRE_BRAZE_INSPECTION",
        }
        for task_id in task_ids:
            task = self.manufacturing_runtime.graph.get(task_id)
            replacement = zones.get(task.task_type)
            if replacement is not None:
                task.required_zones = list(replacement)
            station = stations.get(task.task_type)
            if station is not None:
                task.station_id = station
            if task.task_type is TaskType.PICK_FIN:
                # The V2 physical actor performs any Arm1 tool exchange as the
                # first segment of its non-preemptible fin operation, while
                # Arm3 has a fixed hybrid head. Therefore Arm3's OR branch must
                # not wait for an unrelated Arm1 PREPARE_FIN_TOOL node.
                removed = [
                    predecessor
                    for predecessor in task.predecessors
                    if self.manufacturing_runtime.graph.get(predecessor).task_type
                    is TaskType.PREPARE_FIN_TOOL
                ]
                task.predecessors = [
                    predecessor for predecessor in task.predecessors if predecessor not in removed
                ]
                if removed and not task.predecessors:
                    branch_ready = next(
                        candidate
                        for candidate in (
                            self.manufacturing_runtime.graph.get(candidate_id) for candidate_id in task_ids
                        )
                        if candidate.unit_id == task.unit_id
                        and candidate.task_type is TaskType.CONFIGURE_COMB
                    )
                    task.predecessors.append(branch_ready.task_id)
                    if task.task_id not in branch_ready.successors:
                        branch_ready.successors.append(task.task_id)
                for predecessor in removed:
                    parent = self.manufacturing_runtime.graph.get(predecessor)
                    parent.successors = [
                        successor for successor in parent.successors if successor != task.task_id
                    ]

    def tick(self, dt: float) -> dict[str, Any]:
        self._sync_physical_resources()
        now = float(self.physical_runtime.sim_time)
        self.manufacturing_runtime.tick(now)
        self.physical_runtime.tick(dt)
        now = float(self.physical_runtime.sim_time)
        self.manufacturing_runtime.advance_active_skills(now)
        self.manufacturing_runtime.tick(now, poll_executor=False)
        return self.snapshot()

    @property
    def complete(self) -> bool:
        return self.physical_runtime.complete and self.manufacturing_runtime.terminal

    def pause(self) -> None:
        self.physical_runtime.pause()
        self.manufacturing_runtime.pause(self.sim_time)

    def continue_run(self) -> None:
        self.physical_runtime.continue_run()
        self.manufacturing_runtime.resume(self.sim_time)

    def reset(self) -> None:
        self.physical_runtime.reset()
        self.manufacturing_runtime.reset(self.sim_time)
        self.physical_runtime.set_execution_gate(self.bridge)

    @staticmethod
    def _completion_check(passed: bool, success: str, failure: str) -> dict[str, Any]:
        return {
            "passed": bool(passed),
            "reason_zh": success if passed else failure,
        }

    def physical_completion_report(self) -> dict[str, Any]:
        """One fail-closed authority for the final physical completion claim."""

        manufacturing = self.manufacturing_runtime.snapshot(self.sim_time)
        tasks = list(manufacturing.get("tasks", ()))
        task_outcomes_safe = bool(tasks) and all(
            task.get("status") in {"SUCCEEDED", "CANCELLED"} for task in tasks
        )
        reservations_released = not manufacturing.get("space_time_reservations")
        physical_gate = self.bridge.physical_gate
        scene_callback = getattr(physical_gate, "physical_terminal_gate_snapshot", None)
        scene = scene_callback() if callable(scene_callback) else {}
        logical_trays_safe = bool(self.physical_runtime.flow.trays) and all(
            tray.owner is TrayOwner.EMPTY_BUFFER and tray.order_id is None and tray.unit_id is None
            for tray in self.physical_runtime.flow.trays
        )
        furnace = self.physical_runtime.furnace.state
        logical_furnace_safe = bool(
            not furnace.front_door_open
            and not furnace.rear_door_open
            and all(layer.tray_id is None and not layer.locked for layer in furnace.layers)
            and not self.physical_runtime.output_gate_open
        )
        physical_faults_resolved = all(
            record.recovered for record in self.physical_runtime.faults.faults.values()
        ) and all(plan.status.value == "SUCCEEDED" for plan in self.physical_runtime.faults.plans.values())
        manufacturing_faults_resolved = all(
            fault.recovered for fault in self.manufacturing_runtime.faults.values()
        ) and all(
            plan.status.value == "SUCCEEDED" for plan in self.manufacturing_runtime.recovery.plans.values()
        )
        checks = {
            "manufacturing_terminal": self._completion_check(
                self.manufacturing_runtime.terminal,
                "制造任务图全部进入终态",
                "制造任务图仍有未完成任务",
            ),
            "task_outcomes": self._completion_check(
                task_outcomes_safe,
                "所有任务均成功或安全取消",
                "任务为空或存在失败、阻塞、运行中状态",
            ),
            "physical_runtime_terminal": self._completion_check(
                self.physical_runtime.complete,
                "V2物理actor流程已完成",
                "V2物理actor流程尚未完成",
            ),
            "operations_idle": self._completion_check(
                not self.physical_runtime.operations,
                "无活动物理工序",
                "仍有物理工序正在执行",
            ),
            "tray_transport_settled": self._completion_check(
                bool(scene.get("transport_settled")),
                "全部载件运输已停稳",
                "托盘仍在移动或等待所有权交接",
            ),
            "motion_reservations_released": self._completion_check(
                reservations_released,
                "全部时空预约已释放",
                "仍存在活动时空路径预约",
            ),
            "furnace_and_output_safe": self._completion_check(
                logical_furnace_safe and bool(scene.get("mechanisms_safe")),
                "炉门、升降、推叉和成品门均处于安全终态",
                "炉体或成品出口机构尚未处于安全终态",
            ),
            "tray_ownership_safe": self._completion_check(
                logical_trays_safe and bool(scene.get("physical_trays_safe")),
                "逻辑与物理托盘均回到空托盘缓存",
                "存在未回收托盘或逻辑/物理所有权不一致",
            ),
            "faults_resolved": self._completion_check(
                physical_faults_resolved and manufacturing_faults_resolved,
                "无未恢复故障或人工审核",
                "仍有未恢复故障或人工审核",
            ),
            "robots_settled": self._completion_check(
                bool(scene.get("robots_settled")),
                "三台机械臂均已结束当前物理计划",
                "仍有机械臂运动或执行失败未清除",
            ),
            "physical_gate_bound": self._completion_check(
                physical_gate is not None and callable(scene_callback),
                "统一运行时已绑定真实MuJoCo物理反馈门",
                "未绑定可验证的MuJoCo物理反馈门",
            ),
        }
        failed = [name for name, item in checks.items() if not item["passed"]]
        return {
            "passed": not failed,
            "checks": checks,
            "failed_checks": failed,
            "failed_reasons_zh": [checks[name]["reason_zh"] for name in failed],
            "scene": scene,
        }

    def snapshot(self) -> dict[str, Any]:
        state = self.physical_runtime.snapshot()
        manufacturing = self.manufacturing_runtime.snapshot(self.sim_time)
        state["manufacturing"] = manufacturing
        state["execution_mode"] = "UNIFIED_PHYSICAL_RUNTIME"
        completion = self.physical_completion_report()
        state["physical_completion_gates"] = completion
        state["physical_execution_complete"] = bool(completion["passed"])
        return state


__all__ = ["UnifiedV2Runtime", "V2PhysicalExecutionBridge"]
