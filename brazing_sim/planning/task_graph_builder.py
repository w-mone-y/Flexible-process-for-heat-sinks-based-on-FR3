"""Build executable manufacturing DAGs from validated ``ProcessPlan`` objects."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..flexible.models import ProcessPlan, RackAssignment, RouteStrategy
from .task_graph import TaskGraph
from .task_models import ManufacturingTask, TaskType

DEFAULT_DURATIONS: dict[TaskType, float] = {
    TaskType.INDEX_MATERIAL_KIT: 1.5,
    TaskType.INDEX_EMPTY_TRAY: 2.0,
    TaskType.REMOVE_OLD_PRESS: 2.5,
    TaskType.REMOVE_OLD_COMB: 3.0,
    TaskType.REMOVE_OLD_MOLD: 3.0,
    TaskType.FETCH_MOLD: 2.5,
    TaskType.INSTALL_MOLD: 3.0,
    TaskType.VERIFY_MOLD: 0.5,
    TaskType.VERIFY_CHANGEOVER: 1.2,
    TaskType.PICK_BASE_PLATE: 12.0,
    TaskType.PLACE_BASE_PLATE: 12.0,
    TaskType.VERIFY_BASE_ALIGNMENT: 4.0,
    # Keep the visible quick-change motion smooth enough to overlap the next
    # tray's dispensing/inspection window instead of compressing its final
    # loaded withdrawal into the exact tick when Arm2 becomes available.
    TaskType.PREPARE_FIN_TOOL: 12.0,
    # Four/five/seven fin orders are dispensed at physical path speed.  A
    # 24-second planning envelope also exposes the intended cross-pallet
    # overlap: Arm1 can begin installing the returning tray while Arm2 is
    # still coating the next tray at the process nest.
    TaskType.DISPENSE_BRAZING: 24.0,
    TaskType.INSPECT_BRAZING: 10.0,
    TaskType.CONFIGURE_COMB: 2.0,
    TaskType.FETCH_COMB: 2.5,
    TaskType.INSTALL_COMB: 3.0,
    TaskType.VERIFY_COMB: 0.5,
    TaskType.PICK_FIN: 8.0,
    TaskType.INSTALL_FIN: 10.0,
    TaskType.INSPECT_FINS: 10.0,
    TaskType.FETCH_PRESS_MODULE: 2.0,
    TaskType.INSTALL_PRESS_MODULE: 2.5,
    TaskType.APPLY_PRESS: 2.0,
    TaskType.LOCK_FIXTURE: 1.0,
    TaskType.TRANSFER_S1_S2A: 2.2,
    TaskType.TRANSFER_S2A_S2B: 1.8,
    TaskType.TRANSFER_S2B_S3: 2.2,
    TaskType.TRANSFER_S3_RACK: 2.0,
    TaskType.VERIFY_TRANSFER: 0.25,
    TaskType.ROTATE_TABLE2: 2.0,
    TaskType.VERIFY_TURNTABLE: 0.3,
    TaskType.TRANSFER_TRAY_OUT: 4.0,
    TaskType.MOVE_ELEVATOR: 4.0,
    TaskType.LOAD_RACK_LAYER: 5.0,
    TaskType.LOCK_RACK_LAYER: 1.0,
    TaskType.BATCH_READY: 0.0,
    TaskType.RUN_FURNACE: 10.0,
    TaskType.UNLOAD_RACK_LAYER: 6.0,
    TaskType.POST_BRAZE_INSPECTION: 10.0,
    TaskType.SECOND_POST_BRAZE_VIEW: 6.0,
    # The finished-goods gate, loaded entry, manual payload handoff, empty
    # tray return and gate close are all physically visible stages.
    TaskType.ROUTE_PASS: 6.5,
    TaskType.ROUTE_REWORK: 6.5,
    TaskType.ROUTE_SCRAP: 6.5,
}


def _task_id(order_id: str, unit_index: int, suffix: str) -> str:
    return f"{order_id}_U{unit_index + 1:02d}_{suffix}"


class ProcessPlanTaskGraphBuilder:
    """Generate one graph for every unit plus a shared furnace-batch gate."""

    def __init__(
        self,
        durations: dict[str | TaskType, float] | None = None,
        *,
        flexible_cell: bool = False,
    ) -> None:
        self.durations = dict(DEFAULT_DURATIONS)
        for key, value in (durations or {}).items():
            self.durations[TaskType(key)] = float(value)
        self._sequence = 0
        self.flexible_cell = bool(flexible_cell)

    def _make(
        self,
        *,
        task_id: str,
        task_type: TaskType,
        plan: ProcessPlan,
        unit_id: str,
        tray_id: str | None,
        predecessors: Iterable[str] = (),
        resources: Iterable[str] = (),
        zones: Iterable[str] = (),
        tool: str | None = None,
        retry_limit: int = 0,
        payload: dict[str, Any] | None = None,
        station_id: str | None = None,
        nest_id: str | None = None,
        station_capabilities: Iterable[str] = (),
        route_phase: str | None = None,
        motion_constraints: dict[str, Any] | None = None,
    ) -> ManufacturingTask:
        self._sequence += 1
        return ManufacturingTask(
            task_id=task_id,
            task_type=task_type,
            order_id=plan.order.order_id,
            unit_id=unit_id,
            tray_id=tray_id,
            station_id=station_id,
            nest_id=nest_id,
            station_capabilities=list(station_capabilities),
            route_phase=route_phase,
            motion_constraints=dict(motion_constraints or {}),
            predecessors=list(predecessors),
            eligible_resources=list(resources),
            required_tool=tool,
            required_zones=list(zones),
            estimated_duration=self.durations.get(task_type, 1.0),
            priority=plan.order.priority,
            retry_limit=retry_limit,
            payload=dict(payload or {}),
            sequence_index=self._sequence,
        )

    def _build_unit(
        self,
        graph: TaskGraph,
        plan: ProcessPlan,
        assignment: RackAssignment,
    ) -> str:
        index = assignment.unit_index
        unit_id = f"{plan.order.order_id}_UNIT_{index + 1:02d}"
        tray_id = assignment.tray_id
        prefix = lambda name: _task_id(plan.order.order_id, index, name)  # noqa: E731

        pick_base = self._make(
            task_id=prefix("PICK_BASE"),
            task_type=TaskType.PICK_BASE_PLATE,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            resources=("ARM1",),
            zones=("ZONE_TABLE1",),
            tool="vacuum_gripper",
            retry_limit=1,
        )
        graph.add_task(pick_base)
        place_base = self._make(
            task_id=prefix("PLACE_BASE"),
            task_type=TaskType.PLACE_BASE_PLATE,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(pick_base.task_id,),
            resources=("ARM1",),
            zones=("ZONE_TABLE2_CORE",),
            tool="vacuum_gripper",
            retry_limit=1,
        )
        graph.add_task(place_base)

        # These branches are intentionally independent after base placement:
        # Arm2 can dispense while Arm1 changes to the fin gripper.
        prepare_tool = self._make(
            task_id=prefix("PREPARE_FIN_TOOL"),
            task_type=TaskType.PREPARE_FIN_TOOL,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(place_base.task_id,),
            resources=("ARM1",),
            zones=("ZONE_TOOL_CHANGE",),
            tool="parallel_gripper",
            retry_limit=1,
        )
        graph.add_task(prepare_tool)
        dispense = self._make(
            task_id=prefix("DISPENSE"),
            task_type=TaskType.DISPENSE_BRAZING,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(place_base.task_id,),
            resources=("ARM2",),
            zones=("ZONE_TABLE2_CORE",),
            tool="brazing_dispenser",
            retry_limit=2,
            payload={
                "path_ids": [path.path_id for path in plan.brazing_paths],
                "material_speed_m_s": plan.product.material_speed_m_s,
                "nozzle_tip_height_m": plan.product.nozzle_tip_height_m,
                "nozzle_spacing_m": plan.product.nozzle_spacing_m,
            },
        )
        graph.add_task(dispense)
        inspect_material = self._make(
            task_id=prefix("INSPECT_BRAZING"),
            task_type=TaskType.INSPECT_BRAZING,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(dispense.task_id,),
            resources=("ARM3",),
            zones=("ZONE_TABLE2_CORE",),
            retry_limit=2,
            payload={"path_ids": [path.path_id for path in plan.brazing_paths]},
        )
        graph.add_task(inspect_material)
        configure_comb = self._make(
            task_id=prefix("CONFIGURE_COMB"),
            task_type=TaskType.CONFIGURE_COMB,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(inspect_material.task_id,),
            resources=("FIXTURE",),
            zones=("ZONE_TABLE2_CORE",),
            payload={"comb_module_name": plan.fixture_module.name},
        )
        graph.add_task(configure_comb)

        install_ids: list[str] = []
        previous_install: str | None = None
        for target in plan.fin_targets:
            predecessors = [configure_comb.task_id, prepare_tool.task_id]
            if previous_install is not None:
                predecessors.append(previous_install)
            pick_fin = self._make(
                task_id=prefix(f"PICK_FIN_{target.index + 1:02d}"),
                task_type=TaskType.PICK_FIN,
                plan=plan,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=predecessors,
                resources=("ARM1",),
                zones=("ZONE_TABLE1",),
                tool="parallel_gripper",
                retry_limit=2,
                payload={"fin_id": target.fin_id, "target_position": target.position},
            )
            graph.add_task(pick_fin)
            install_fin = self._make(
                task_id=prefix(f"INSTALL_FIN_{target.index + 1:02d}"),
                task_type=TaskType.INSTALL_FIN,
                plan=plan,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=(pick_fin.task_id,),
                resources=("ARM1",),
                zones=("ZONE_TABLE2_CORE",),
                tool="parallel_gripper",
                retry_limit=2,
                payload={"fin_id": target.fin_id, "target_position": target.position},
            )
            graph.add_task(install_fin)
            previous_install = install_fin.task_id
            install_ids.append(install_fin.task_id)

        inspect_fins = self._make(
            task_id=prefix("INSPECT_FINS"),
            task_type=TaskType.INSPECT_FINS,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=install_ids,
            resources=("ARM3",),
            zones=("ZONE_TABLE2_CORE",),
            retry_limit=2,
            payload={"fin_ids": [target.fin_id for target in plan.fin_targets]},
        )
        graph.add_task(inspect_fins)
        press = self._make(
            task_id=prefix("APPLY_PRESS"),
            task_type=TaskType.APPLY_PRESS,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(inspect_fins.task_id,),
            resources=("FIXTURE",),
            zones=("ZONE_TABLE2_CORE",),
            payload={
                "target_force_n": plan.product.target_clamping_force_n,
                "force_hold_duration_s": plan.product.force_hold_duration_s,
            },
        )
        graph.add_task(press)
        lock_fixture = self._make(
            task_id=prefix("LOCK_FIXTURE"),
            task_type=TaskType.LOCK_FIXTURE,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(press.task_id,),
            resources=("FIXTURE",),
            zones=("ZONE_TABLE2_CORE",),
        )
        graph.add_task(lock_fixture)
        transfer_out = self._make(
            task_id=prefix("TRANSFER_OUT"),
            task_type=TaskType.TRANSFER_TRAY_OUT,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(lock_fixture.task_id,),
            resources=("OUTFEED",),
            zones=("ZONE_OUTFEED",),
            retry_limit=1,
        )
        graph.add_task(transfer_out)
        move_lift = self._make(
            task_id=prefix("MOVE_ELEVATOR"),
            task_type=TaskType.MOVE_ELEVATOR,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(transfer_out.task_id,),
            resources=("ELEVATOR",),
            zones=("ZONE_ELEVATOR_TRANSFER",),
            retry_limit=1,
            payload={"layer_index": assignment.layer_index, "height_m": assignment.height_m},
        )
        graph.add_task(move_lift)
        load_layer = self._make(
            task_id=prefix("LOAD_RACK"),
            task_type=TaskType.LOAD_RACK_LAYER,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(move_lift.task_id,),
            resources=("TRANSFER_FORK",),
            zones=("ZONE_RACK_FRONT", "ZONE_FURNACE_LOADING"),
            retry_limit=1,
            payload={"layer_index": assignment.layer_index, "height_m": assignment.height_m},
        )
        graph.add_task(load_layer)
        lock_layer = self._make(
            task_id=prefix("LOCK_RACK"),
            task_type=TaskType.LOCK_RACK_LAYER,
            plan=plan,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(load_layer.task_id,),
            resources=(f"RACK_LAYER_{assignment.layer_index + 1:02d}",),
            zones=("ZONE_RACK_FRONT",),
            payload={"layer_index": assignment.layer_index, "height_m": assignment.height_m},
        )
        graph.add_task(lock_layer)
        return lock_layer.task_id

    @staticmethod
    def _replace_zone(task: ManufacturingTask, old: str, new: str) -> None:
        task.required_zones = [new if zone == old else zone for zone in task.required_zones]

    def _decorate_async_line(self, graph: TaskGraph, plan: ProcessPlan) -> None:
        """Route every unit through the four independent shallow-U stations.

        The previous flexible-cell decorator modelled a synchronous two-nest
        turntable.  The current line deliberately has no global index event:
        each pallet owns one station or one transfer slide, and advances as
        soon as its downstream station becomes free.  The explicit transfer
        nodes are also the station-release events used by later units.
        """

        order_id = plan.order.order_id
        lookups: list[dict[str, ManufacturingTask]] = []
        transfer_ids: list[dict[str, str]] = []

        def station_task(
            task: ManufacturingTask,
            station_id: str,
            zone: str,
            capability: str,
            phase: str,
        ) -> None:
            self._replace_zone(task, "ZONE_TABLE2_CORE", zone)
            task.station_id = station_id
            task.nest_id = None
            task.station_capabilities = [capability]
            task.route_phase = phase
            task.motion_constraints.setdefault("minimum_clearance_m", 0.04)
            task.motion_constraints.setdefault("sample_interval_m", 0.01)
            task.motion_constraints.setdefault("time_sample_s", 0.02)

        for index, assignment in enumerate(plan.rack_assignments):
            prefix = f"{order_id}_U{index + 1:02d}_"
            items = {
                task.task_id.removeprefix(prefix): task for task in graph if task.task_id.startswith(prefix)
            }
            lookups.append(items)
            unit_id = f"{order_id}_UNIT_{index + 1:02d}"
            tray_id = assignment.tray_id

            index_tray = self._make(
                task_id=f"{prefix}INDEX_TRAY",
                task_type=TaskType.INDEX_EMPTY_TRAY,
                plan=plan,
                unit_id=unit_id,
                tray_id=tray_id,
                resources=("EMPTY_TRAY_INDEXER",),
                zones=("ZONE_S1_ARM1",),
                station_id="S1_BASE_LOADING",
                station_capabilities=("BASE_LOADING",),
                route_phase="EMPTY",
            )
            graph.add_task(index_tray)
            graph.add_dependency(index_tray.task_id, items["PICK_BASE"].task_id)

            station_task(
                items["PLACE_BASE"],
                "S1_BASE_LOADING",
                "ZONE_S1_ARM1",
                "BASE_LOADING",
                "AT_S1",
            )
            items["PICK_BASE"].required_zones = ["ZONE_BASE_MAGAZINE"]
            items["PICK_BASE"].motion_constraints["lock_joint_indices"] = [6]

            station_task(
                items["DISPENSE"],
                "S2A_DISPENSING",
                "ZONE_S2A_ARM2",
                "BRAZING",
                "AT_S2A",
            )
            items["DISPENSE"].motion_constraints["tool_z_vertical_tolerance_deg"] = 0.1
            station_task(
                items["INSPECT_BRAZING"],
                "S2B_MATERIAL_INSPECTION",
                "ZONE_S2B_ARM3",
                "MATERIAL_INSPECTION",
                "AT_S2B",
            )

            for suffix, task in items.items():
                if task.task_type in {
                    TaskType.CONFIGURE_COMB,
                    TaskType.INSTALL_FIN,
                    TaskType.INSPECT_FINS,
                    TaskType.APPLY_PRESS,
                    TaskType.LOCK_FIXTURE,
                }:
                    station_task(
                        task,
                        "S3_FIN_ASSEMBLY",
                        "ZONE_S3_SHARED",
                        "FIN_ASSEMBLY",
                        "AT_S3",
                    )
                if task.task_type in {TaskType.PICK_FIN, TaskType.INSTALL_FIN}:
                    task.motion_constraints["lock_joint_indices"] = [6]
                if task.task_type is TaskType.PICK_FIN:
                    task.required_zones = ["ZONE_FIN_MAGAZINE"]
                elif task.task_type is TaskType.INSTALL_FIN:
                    # Arm1's S3 insertion posture and Arm3's finished-output
                    # camera posture have overlapping link-6/7 swept volumes
                    # even though their nominal stations are different.  A
                    # shared reservation serialises only these two hazardous
                    # motions; base loading, dispensing and other inspections
                    # remain free to overlap.
                    task.required_zones.append("ZONE_S3_OUTPUT_INTERARM")

            # S1 -> S2A.  A high-reliability route performs its extra base
            # view at S1 before the pallet is released to the first slide.
            transfer_12_predecessor = items["PLACE_BASE"].task_id
            high_reliability = plan.route_strategy is RouteStrategy.HIGH_RELIABILITY or (
                plan.route_strategy is RouteStrategy.FIRST_ARTICLE and index == 0
            )
            if high_reliability:
                verify_base = self._make(
                    task_id=f"{prefix}VERIFY_BASE_ALIGNMENT",
                    task_type=TaskType.VERIFY_BASE_ALIGNMENT,
                    plan=plan,
                    unit_id=unit_id,
                    tray_id=tray_id,
                    predecessors=(items["PLACE_BASE"].task_id,),
                    resources=("ARM3",),
                    zones=("ZONE_S1_ARM1",),
                    station_id="S1_BASE_LOADING",
                    station_capabilities=("BASE_ALIGNMENT",),
                    route_phase="AT_S1",
                    payload={"second_confirmation": True},
                )
                graph.add_task(verify_base)
                transfer_12_predecessor = verify_base.task_id

            transfer_12 = self._make(
                task_id=f"{prefix}TRANSFER_S1_S2A",
                task_type=TaskType.TRANSFER_S1_S2A,
                plan=plan,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=(transfer_12_predecessor,),
                resources=("TRANSFER_S1_S2A",),
                # A loaded slide owns both endpoint junctions until the
                # carriage has arrived, handed the tray off and returned to
                # zero.  Releasing S1 as soon as the logical owner changes
                # allowed the next pallet to enter the same swept volume.
                zones=("ZONE_TRANSFER_12", "ZONE_S1_ARM1", "ZONE_S2A_ARM2"),
                route_phase="TRANSFER_S1_S2A",
                payload={"source_station": "S1", "target_station": "S2A"},
            )
            graph.add_task(transfer_12)
            graph.remove_dependency(items["PLACE_BASE"].task_id, items["DISPENSE"].task_id)
            graph.add_dependency(transfer_12.task_id, items["DISPENSE"].task_id)

            transfer_2a_2b = self._make(
                task_id=f"{prefix}TRANSFER_S2A_S2B",
                task_type=TaskType.TRANSFER_S2A_S2B,
                plan=plan,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=(items["DISPENSE"].task_id,),
                resources=("TRANSFER_S2A_S2B",),
                zones=("ZONE_TRANSFER_2A_2B", "ZONE_S2A_ARM2", "ZONE_S2B_ARM3"),
                route_phase="TRANSFER_S2A_S2B",
                payload={"source_station": "S2A", "target_station": "S2B"},
            )
            graph.add_task(transfer_2a_2b)
            graph.remove_dependency(items["DISPENSE"].task_id, items["INSPECT_BRAZING"].task_id)
            graph.add_dependency(transfer_2a_2b.task_id, items["INSPECT_BRAZING"].task_id)

            transfer_2b_3 = self._make(
                task_id=f"{prefix}TRANSFER_S2B_S3",
                task_type=TaskType.TRANSFER_S2B_S3,
                plan=plan,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=(items["INSPECT_BRAZING"].task_id,),
                resources=("TRANSFER_S2B_S3",),
                zones=("ZONE_TRANSFER_23", "ZONE_S2B_ARM3", "ZONE_S3_SHARED"),
                route_phase="TRANSFER_S2B_S3",
                payload={"source_station": "S2B", "target_station": "S3"},
            )
            graph.add_task(transfer_2b_3)
            graph.remove_dependency(items["INSPECT_BRAZING"].task_id, items["CONFIGURE_COMB"].task_id)
            graph.add_dependency(transfer_2b_3.task_id, items["CONFIGURE_COMB"].task_id)

            transfer_3_rack = items["TRANSFER_OUT"]
            transfer_3_rack.task_type = TaskType.TRANSFER_S3_RACK
            transfer_3_rack.eligible_resources = ["TRANSFER_S3_RACK"]
            transfer_3_rack.required_zones = [
                "ZONE_TRANSFER_3_RACK",
                "ZONE_S3_SHARED",
                "ZONE_RACK_FRONT",
            ]
            transfer_3_rack.station_id = None
            transfer_3_rack.route_phase = "TRANSFER_S3_RACK"
            transfer_3_rack.payload.update({"source_station": "S3", "target_station": "RACK_INFEED"})
            transfer_ids.append(
                {
                    "12": transfer_12.task_id,
                    "2a2b": transfer_2a_2b.task_id,
                    "2b3": transfer_2b_3.task_id,
                    "3r": transfer_3_rack.task_id,
                }
            )

        # Station-capacity dependencies form the pipeline.  They release a
        # station at the moment the previous pallet has physically left it,
        # not when the whole product is complete.
        for index in range(1, len(lookups)):
            graph.add_dependency(transfer_ids[index - 1]["12"], f"{order_id}_U{index + 1:02d}_INDEX_TRAY")
            graph.add_dependency(transfer_ids[index - 1]["2a2b"], transfer_ids[index]["12"])
            graph.add_dependency(transfer_ids[index - 1]["2b3"], transfer_ids[index]["2a2b"])
            graph.add_dependency(transfer_ids[index - 1]["3r"], transfer_ids[index]["2b3"])

    def _decorate_flexible_cell(self, graph: TaskGraph, plan: ProcessPlan) -> None:
        """Turn the safe legacy unit chains into a two-nest physical pipeline.

        One rotation node represents both nests.  For boundary ``i`` it returns
        unit ``i-1`` from PROCESS to ASSEMBLY and simultaneously sends unit
        ``i`` from ASSEMBLY to PROCESS.  This prevents the common but invalid
        model where the two trays appear to rotate independently.
        """

        order_id = plan.order.order_id
        quantity = len(plan.rack_assignments)

        def uses_high_reliability_route(index: int) -> bool:
            return plan.route_strategy is RouteStrategy.HIGH_RELIABILITY or (
                plan.route_strategy is RouteStrategy.FIRST_ARTICLE and index == 0
            )

        lookup: list[dict[str, ManufacturingTask]] = []
        for index in range(quantity):
            prefix = f"{order_id}_U{index + 1:02d}_"
            items = {
                task.task_id.removeprefix(prefix): task for task in graph if task.task_id.startswith(prefix)
            }
            lookup.append(items)
            nest_id = "NEST_A" if index % 2 == 0 else "NEST_B"
            assembly_types = {
                TaskType.PLACE_BASE_PLATE,
                TaskType.CONFIGURE_COMB,
                TaskType.INSTALL_FIN,
                TaskType.INSPECT_FINS,
                TaskType.APPLY_PRESS,
                TaskType.LOCK_FIXTURE,
            }
            process_types = {TaskType.DISPENSE_BRAZING, TaskType.INSPECT_BRAZING}
            for task in items.values():
                task.nest_id = nest_id
                if task.task_type in assembly_types:
                    self._replace_zone(task, "ZONE_TABLE2_CORE", "ZONE_TABLE2_ASSEMBLY")
                    task.station_id = "TABLE2_ASSEMBLY"
                    task.station_capabilities = ["ASSEMBLY"]
                elif task.task_type in process_types:
                    self._replace_zone(task, "ZONE_TABLE2_CORE", "ZONE_TABLE2_PROCESS")
                    task.station_id = "TABLE2_PROCESS"
                    task.station_capabilities = ["BRAZING", "MATERIAL_INSPECTION"]
                if task.task_type in {TaskType.PICK_BASE_PLATE, TaskType.PLACE_BASE_PLATE}:
                    task.route_phase = "MOLD_READY"
                elif task.task_type in process_types:
                    task.route_phase = "BASE_READY"
                elif task.task_type in {
                    TaskType.CONFIGURE_COMB,
                    TaskType.PICK_FIN,
                    TaskType.INSTALL_FIN,
                    TaskType.INSPECT_FINS,
                }:
                    task.route_phase = "MATERIAL_READY"
                elif task.task_type in {TaskType.APPLY_PRESS, TaskType.LOCK_FIXTURE}:
                    task.route_phase = "ASSEMBLY_READY"
                task.motion_constraints.setdefault("minimum_clearance_m", 0.04)
                task.motion_constraints.setdefault("sample_interval_m", 0.01)
                task.motion_constraints.setdefault("time_sample_s", 0.02)
                if task.task_type in {TaskType.PICK_BASE_PLATE, TaskType.PICK_FIN, TaskType.INSTALL_FIN}:
                    task.motion_constraints["lock_joint_indices"] = [6]
                if task.task_type is TaskType.DISPENSE_BRAZING:
                    task.motion_constraints["tool_z_vertical_tolerance_deg"] = 0.1

        # Visible old-module removal and mold installation precede each base.
        for index, items in enumerate(lookup):
            unit_id = f"{order_id}_UNIT_{index + 1:02d}"
            tray_id = plan.rack_assignments[index].tray_id
            nest_id = "NEST_A" if index % 2 == 0 else "NEST_B"
            predecessor: tuple[str, ...] = ()
            if index >= 2:
                predecessor = (lookup[index - 2]["TRANSFER_OUT"].task_id,)
            chain: list[tuple[str, TaskType, str]] = [
                ("INDEX_TRAY", TaskType.INDEX_EMPTY_TRAY, "EMPTY_TRAY_INDEXER"),
                ("REMOVE_PRESS", TaskType.REMOVE_OLD_PRESS, "CHANGEOVER_GANTRY"),
                ("REMOVE_COMB", TaskType.REMOVE_OLD_COMB, "CHANGEOVER_GANTRY"),
                ("REMOVE_MOLD", TaskType.REMOVE_OLD_MOLD, "CHANGEOVER_GANTRY"),
                ("FETCH_MOLD", TaskType.FETCH_MOLD, "CHANGEOVER_GANTRY"),
                ("INSTALL_MOLD", TaskType.INSTALL_MOLD, "CHANGEOVER_GANTRY"),
                ("VERIFY_MOLD", TaskType.VERIFY_MOLD, "CHANGEOVER_GANTRY"),
            ]
            previous = predecessor
            for suffix, task_type, resource in chain:
                task = self._make(
                    task_id=f"{order_id}_U{index + 1:02d}_{suffix}",
                    task_type=task_type,
                    plan=plan,
                    unit_id=unit_id,
                    tray_id=tray_id,
                    predecessors=previous,
                    resources=(resource,),
                    zones=(
                        "ZONE_TABLE2_ASSEMBLY",
                        "ZONE_CHANGEOVER_GANTRY",
                    ),
                    station_id="TABLE2_ASSEMBLY",
                    nest_id=nest_id,
                    route_phase="CHANGEOVER",
                    payload={"module_name": plan.fixture_module.name},
                )
                graph.add_task(task)
                previous = (task.task_id,)
            graph.add_dependency(previous[0], items["PICK_BASE"].task_id)

            # Material-kit selection shares no swept volume with the gantry and
            # can therefore run while the mold is being fitted.
            kit = self._make(
                task_id=f"{order_id}_U{index + 1:02d}_INDEX_MATERIAL",
                task_type=TaskType.INDEX_MATERIAL_KIT,
                plan=plan,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=predecessor,
                resources=("MATERIAL_INDEXER",),
                zones=("ZONE_MATERIAL_INDEXER",),
                payload={"product_id": plan.product.product_id},
            )
            graph.add_task(kit)
            graph.add_dependency(kit.task_id, items["PICK_BASE"].task_id)

            if uses_high_reliability_route(index):
                verify_mold = graph.get(f"{order_id}_U{index + 1:02d}_VERIFY_MOLD")
                graph.remove_dependency(verify_mold.task_id, items["PICK_BASE"].task_id)
                verify_changeover = self._make(
                    task_id=f"{order_id}_U{index + 1:02d}_VERIFY_CHANGEOVER",
                    task_type=TaskType.VERIFY_CHANGEOVER,
                    plan=plan,
                    unit_id=unit_id,
                    tray_id=tray_id,
                    predecessors=(verify_mold.task_id,),
                    resources=("CHANGEOVER_GANTRY",),
                    zones=("ZONE_TABLE2_ASSEMBLY", "ZONE_CHANGEOVER_GANTRY"),
                    station_id="TABLE2_ASSEMBLY",
                    nest_id=nest_id,
                    route_phase="CHANGEOVER",
                    payload={
                        "module_name": plan.fixture_module.name,
                        "second_confirmation": True,
                    },
                )
                graph.add_task(verify_changeover)
                graph.add_dependency(verify_changeover.task_id, items["PICK_BASE"].task_id)

        # One physical turntable task advances both nests at every boundary.
        swap_verifies: list[str] = []
        for index, items in enumerate(lookup):
            if index == 0:
                predecessors = [items["PLACE_BASE"].task_id]
            else:
                predecessors = [lookup[index - 1]["INSPECT_BRAZING"].task_id, items["PLACE_BASE"].task_id]
            rotate = self._make(
                task_id=f"{order_id}_SWAP_{index + 1:02d}",
                task_type=TaskType.ROTATE_TABLE2,
                plan=plan,
                unit_id=f"{order_id}_TURNTABLE",
                tray_id=None,
                predecessors=predecessors,
                resources=("TURNTABLE",),
                zones=("ZONE_TABLE2_ASSEMBLY", "ZONE_TABLE2_PROCESS", "ZONE_TURNTABLE_SWEEP"),
                station_capabilities=("SYNCHRONOUS_NEST_SWAP",),
                motion_constraints={"s_curve": True, "rotation_deg": 180.0, "settle_s": 0.3},
                payload={
                    "assembly_tray": items["PLACE_BASE"].tray_id,
                    "process_tray": None if index == 0 else lookup[index - 1]["PLACE_BASE"].tray_id,
                },
            )
            graph.add_task(rotate)
            if uses_high_reliability_route(index):
                graph.remove_dependency(items["PLACE_BASE"].task_id, rotate.task_id)
                base_verify = self._make(
                    task_id=f"{order_id}_U{index + 1:02d}_VERIFY_BASE_ALIGNMENT",
                    task_type=TaskType.VERIFY_BASE_ALIGNMENT,
                    plan=plan,
                    unit_id=items["PLACE_BASE"].unit_id,
                    tray_id=items["PLACE_BASE"].tray_id,
                    predecessors=(items["PLACE_BASE"].task_id,),
                    resources=("ARM3",),
                    zones=("ZONE_TABLE2_ASSEMBLY",),
                    station_id="TABLE2_ASSEMBLY",
                    nest_id=items["PLACE_BASE"].nest_id,
                    route_phase="BASE_READY",
                    payload={"second_confirmation": True},
                )
                graph.add_task(base_verify)
                graph.add_dependency(base_verify.task_id, rotate.task_id)
            verify = self._make(
                task_id=f"{order_id}_SWAP_{index + 1:02d}_VERIFY",
                task_type=TaskType.VERIFY_TURNTABLE,
                plan=plan,
                unit_id=f"{order_id}_TURNTABLE",
                tray_id=None,
                predecessors=(rotate.task_id,),
                resources=("TURNTABLE",),
                zones=("ZONE_TURNTABLE_SWEEP",),
                station_capabilities=("SYNCHRONOUS_NEST_SWAP",),
            )
            graph.add_task(verify)
            swap_verifies.append(verify.task_id)
            graph.remove_dependency(items["PLACE_BASE"].task_id, items["DISPENSE"].task_id)
            graph.add_dependency(verify.task_id, items["DISPENSE"].task_id)
            if index > 0:
                previous_comb = lookup[index - 1]["CONFIGURE_COMB"]
                graph.remove_dependency(
                    lookup[index - 1]["INSPECT_BRAZING"].task_id,
                    previous_comb.task_id,
                )
                graph.add_dependency(verify.task_id, previous_comb.task_id)

        if quantity > 1:
            graph.add_dependency(swap_verifies[0], f"{order_id}_U02_INDEX_TRAY")
            graph.add_dependency(swap_verifies[0], f"{order_id}_U02_INDEX_MATERIAL")

        # Return the final process tray after the preceding assembly tray has
        # physically cleared the left nest.
        last_index = quantity - 1
        last = lookup[last_index]
        final_predecessors = [last["INSPECT_BRAZING"].task_id]
        if quantity > 1:
            final_predecessors.append(lookup[last_index - 1]["TRANSFER_OUT"].task_id)
        final_rotate = self._make(
            task_id=f"{order_id}_SWAP_RETURN_FINAL",
            task_type=TaskType.ROTATE_TABLE2,
            plan=plan,
            unit_id=f"{order_id}_TURNTABLE",
            tray_id=None,
            predecessors=final_predecessors,
            resources=("TURNTABLE",),
            zones=("ZONE_TABLE2_ASSEMBLY", "ZONE_TABLE2_PROCESS", "ZONE_TURNTABLE_SWEEP"),
            station_capabilities=("SYNCHRONOUS_NEST_SWAP",),
            motion_constraints={"s_curve": True, "rotation_deg": 180.0, "settle_s": 0.3},
        )
        graph.add_task(final_rotate)
        final_verify = self._make(
            task_id=f"{order_id}_SWAP_RETURN_FINAL_VERIFY",
            task_type=TaskType.VERIFY_TURNTABLE,
            plan=plan,
            unit_id=f"{order_id}_TURNTABLE",
            tray_id=None,
            predecessors=(final_rotate.task_id,),
            resources=("TURNTABLE",),
            zones=("ZONE_TURNTABLE_SWEEP",),
        )
        graph.add_task(final_verify)
        graph.remove_dependency(last["INSPECT_BRAZING"].task_id, last["CONFIGURE_COMB"].task_id)
        graph.add_dependency(final_verify.task_id, last["CONFIGURE_COMB"].task_id)

        # Comb and short press beams are now explicit visible gantry branches.
        for index, items in enumerate(lookup):
            configure = items["CONFIGURE_COMB"]
            original_predecessors = list(configure.predecessors)
            for predecessor_id in original_predecessors:
                graph.remove_dependency(predecessor_id, configure.task_id)
            previous = original_predecessors
            for suffix, task_type in (
                ("FETCH_COMB", TaskType.FETCH_COMB),
                ("INSTALL_COMB", TaskType.INSTALL_COMB),
                ("VERIFY_COMB", TaskType.VERIFY_COMB),
            ):
                task = self._make(
                    task_id=f"{order_id}_U{index + 1:02d}_{suffix}",
                    task_type=task_type,
                    plan=plan,
                    unit_id=configure.unit_id,
                    tray_id=configure.tray_id,
                    predecessors=previous,
                    resources=("CHANGEOVER_GANTRY",),
                    zones=("ZONE_TABLE2_ASSEMBLY", "ZONE_CHANGEOVER_GANTRY"),
                    station_id="TABLE2_ASSEMBLY",
                    nest_id=configure.nest_id,
                    route_phase="MATERIAL_READY",
                    payload={"module_name": plan.fixture_module.name},
                )
                graph.add_task(task)
                previous = [task.task_id]
            graph.add_dependency(previous[0], configure.task_id)

            inspect_fins = items["INSPECT_FINS"]
            press = items["APPLY_PRESS"]
            graph.remove_dependency(inspect_fins.task_id, press.task_id)
            fetch_press = self._make(
                task_id=f"{order_id}_U{index + 1:02d}_FETCH_PRESS",
                task_type=TaskType.FETCH_PRESS_MODULE,
                plan=plan,
                unit_id=press.unit_id,
                tray_id=press.tray_id,
                predecessors=(inspect_fins.task_id,),
                resources=("CHANGEOVER_GANTRY",),
                zones=("ZONE_CHANGEOVER_GANTRY",),
                station_id="TABLE2_ASSEMBLY",
                nest_id=press.nest_id,
                route_phase="ASSEMBLY_READY",
            )
            graph.add_task(fetch_press)
            install_press = self._make(
                task_id=f"{order_id}_U{index + 1:02d}_INSTALL_PRESS",
                task_type=TaskType.INSTALL_PRESS_MODULE,
                plan=plan,
                unit_id=press.unit_id,
                tray_id=press.tray_id,
                predecessors=(fetch_press.task_id,),
                resources=("CHANGEOVER_GANTRY",),
                zones=("ZONE_TABLE2_ASSEMBLY", "ZONE_CHANGEOVER_GANTRY"),
                station_id="TABLE2_ASSEMBLY",
                nest_id=press.nest_id,
                route_phase="ASSEMBLY_READY",
            )
            graph.add_task(install_press)
            graph.add_dependency(install_press.task_id, press.task_id)

    def build(self, plan: ProcessPlan) -> TaskGraph:
        self._sequence = 0
        graph = TaskGraph()
        lock_tasks = [self._build_unit(graph, plan, assignment) for assignment in plan.rack_assignments]
        if self.flexible_cell:
            self._decorate_async_line(graph, plan)
        batch_id = plan.order.order_id
        batch_ready = self._make(
            task_id=f"{batch_id}_BATCH_READY",
            task_type=TaskType.BATCH_READY,
            plan=plan,
            unit_id=f"{batch_id}_BATCH",
            tray_id=None,
            predecessors=lock_tasks,
            resources=("FURNACE",),
            zones=("ZONE_FURNACE_LOADING",),
        )
        graph.add_task(batch_ready)
        furnace = self._make(
            task_id=f"{batch_id}_RUN_FURNACE",
            task_type=TaskType.RUN_FURNACE,
            plan=plan,
            unit_id=f"{batch_id}_BATCH",
            tray_id=None,
            predecessors=(batch_ready.task_id,),
            resources=("FURNACE",),
            zones=("ZONE_FURNACE_LOADING",),
            payload={"recipe": plan.recipe.name, "duration_s": plan.recipe.to_domain().process_seconds},
        )
        graph.add_task(furnace)

        post_inspection_zones = (
            ("ZONE_OUTFEED", "ZONE_S3_OUTPUT_INTERARM") if self.flexible_cell else ("ZONE_OUTFEED",)
        )
        previous_unload: str | None = None
        for assignment in sorted(plan.rack_assignments, key=lambda item: item.layer_index, reverse=True):
            unit_id = f"{plan.order.order_id}_UNIT_{assignment.unit_index + 1:02d}"
            predecessors = [furnace.task_id]
            if previous_unload is not None:
                predecessors.append(previous_unload)
            unload = self._make(
                task_id=_task_id(plan.order.order_id, assignment.unit_index, "UNLOAD_RACK"),
                task_type=TaskType.UNLOAD_RACK_LAYER,
                plan=plan,
                unit_id=unit_id,
                tray_id=assignment.tray_id,
                predecessors=predecessors,
                resources=("ELEVATOR", "TRANSFER_FORK"),
                zones=("ZONE_RACK_FRONT", "ZONE_ELEVATOR_TRANSFER"),
                retry_limit=1,
                payload={"layer_index": assignment.layer_index, "height_m": assignment.height_m},
            )
            graph.add_task(unload)
            inspect = self._make(
                task_id=_task_id(plan.order.order_id, assignment.unit_index, "POST_INSPECT"),
                task_type=TaskType.POST_BRAZE_INSPECTION,
                plan=plan,
                unit_id=unit_id,
                tray_id=assignment.tray_id,
                predecessors=(unload.task_id,),
                resources=("ARM3",),
                zones=post_inspection_zones,
            )
            graph.add_task(inspect)
            route_predecessor = inspect.task_id
            high_reliability = plan.route_strategy is RouteStrategy.HIGH_RELIABILITY or (
                plan.route_strategy is RouteStrategy.FIRST_ARTICLE and assignment.unit_index == 0
            )
            if high_reliability:
                second_view = self._make(
                    task_id=_task_id(
                        plan.order.order_id,
                        assignment.unit_index,
                        "SECOND_POST_VIEW",
                    ),
                    task_type=TaskType.SECOND_POST_BRAZE_VIEW,
                    plan=plan,
                    unit_id=unit_id,
                    tray_id=assignment.tray_id,
                    predecessors=(inspect.task_id,),
                    resources=("ARM3",),
                    zones=post_inspection_zones,
                    payload={"camera_view": "side", "second_confirmation": True},
                )
                graph.add_task(second_view)
                route_predecessor = second_view.task_id
            remove_comb = self._make(
                task_id=_task_id(
                    plan.order.order_id,
                    assignment.unit_index,
                    "REMOVE_FINISHED_COMB",
                ),
                task_type=TaskType.REMOVE_OLD_COMB,
                plan=plan,
                unit_id=unit_id,
                tray_id=assignment.tray_id,
                predecessors=(route_predecessor,),
                resources=("FIXTURE",),
                zones=("ZONE_OUTFEED",),
                payload={"after_brazing": True, "condition_passthrough": True},
            )
            graph.add_task(remove_comb)
            remove_press = self._make(
                task_id=_task_id(
                    plan.order.order_id,
                    assignment.unit_index,
                    "REMOVE_FINISHED_PRESS",
                ),
                task_type=TaskType.REMOVE_OLD_PRESS,
                plan=plan,
                unit_id=unit_id,
                tray_id=assignment.tray_id,
                predecessors=(remove_comb.task_id,),
                resources=("FIXTURE",),
                zones=("ZONE_OUTFEED",),
                payload={"after_brazing": True, "condition_passthrough": True},
            )
            graph.add_task(remove_press)
            route_predecessor = remove_press.task_id
            for disposition, task_type in (
                ("PASS", TaskType.ROUTE_PASS),
                ("REWORK_REQUIRED", TaskType.ROUTE_REWORK),
                ("SCRAPPED", TaskType.ROUTE_SCRAP),
            ):
                route = self._make(
                    task_id=_task_id(plan.order.order_id, assignment.unit_index, f"ROUTE_{disposition}"),
                    task_type=task_type,
                    plan=plan,
                    unit_id=unit_id,
                    tray_id=assignment.tray_id,
                    predecessors=(route_predecessor,),
                    resources=("OUTFEED",),
                    zones=("ZONE_OUTFEED",),
                    payload={"condition": disposition},
                )
                graph.add_task(route)
            previous_unload = unload.task_id
        if self.flexible_cell:
            # The simplified furnace layout has one physical straight-belt
            # carriage.  Legacy task names (elevator/fork/rack lock) remain in
            # the public DAG, but they must reserve that same actuator owner
            # so two pallets can never command the axis concurrently.
            serial_logistics = {
                TaskType.MOVE_ELEVATOR,
                TaskType.LOAD_RACK_LAYER,
                TaskType.LOCK_RACK_LAYER,
                TaskType.UNLOAD_RACK_LAYER,
            }
            for task in graph:
                if task.task_type in serial_logistics:
                    task.eligible_resources = ["OUTFEED"]
        graph.validate_acyclic()
        graph.refresh_ready(0.0)
        return graph


def build_task_graph(
    plan: ProcessPlan,
    durations: dict[str | TaskType, float] | None = None,
    *,
    flexible_cell: bool = False,
) -> TaskGraph:
    return ProcessPlanTaskGraphBuilder(durations, flexible_cell=flexible_cell).build(plan)


__all__ = ["DEFAULT_DURATIONS", "ProcessPlanTaskGraphBuilder", "build_task_graph"]
