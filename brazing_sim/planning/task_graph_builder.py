"""Build executable manufacturing DAGs from validated ``ProcessPlan`` objects.

Durations and resource eligibility are data-driven (steps A and B).  When a
:class:`~brazing_sim.flexible.capability_models.CapabilityCatalog` is available
the builder evaluates each capability's ``duration_model`` against the plan's
real parameters and derives ``eligible_resources`` from capability declarations
instead of naming one arm.  ``LEGACY_DURATIONS`` is retained only as the offline
fallback used when no catalog/resource configuration is supplied.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from ..flexible.capability_loader import load_capabilities, load_routing
from ..flexible.capability_models import CapabilityCatalog, RoutingSpec
from ..flexible.models import ProcessPlan, RackAssignment, RouteStrategy
from ..changeover.config_diff import (
    FixtureConfiguration,
    plan_changeover,
    required_configuration,
)
from ..flexible.routing_compiler import CompiledOperation, RoutingCompiler
from ..paths import CONFIG_DIR
from .capability_binding import (
    UNRESTRICTED_PROFILE,
    CapabilityBinder,
    LineExecutionProfile,
)
from .task_graph import TaskGraph
from .task_models import ManufacturingTask, TaskType

DEFAULT_CAPABILITIES_PATH = CONFIG_DIR / "capabilities.yaml"
DEFAULT_ROUTING_PATH = CONFIG_DIR / "routings" / "heat_sink_standard.yaml"


@lru_cache(maxsize=4)
def _cached_catalog(path: str) -> CapabilityCatalog:
    return load_capabilities(path)


@lru_cache(maxsize=4)
def _cached_routing(path: str, catalog_path: str) -> RoutingSpec:
    return load_routing(path, catalog=_cached_catalog(catalog_path))


def default_capability_catalog() -> CapabilityCatalog:
    """Load (and memoize) the repository's capability ontology."""

    return _cached_catalog(str(DEFAULT_CAPABILITIES_PATH))


def default_routing() -> RoutingSpec:
    """Load (and memoize) the standard heat-sink routing."""

    return _cached_routing(str(DEFAULT_ROUTING_PATH), str(DEFAULT_CAPABILITIES_PATH))


# Nominal task durations used only when no capability catalog is provided.  The
# authoritative values now live in ``config/capabilities.yaml`` as parametric
# ``duration_model`` expressions.
LEGACY_DURATIONS: dict[TaskType, float] = {
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
    TaskType.REVIEW_BRAZING_CLOSEUP: 6.0,
    TaskType.CONFIGURE_COMB: 2.0,
    TaskType.FETCH_COMB: 2.5,
    TaskType.INSTALL_COMB: 3.0,
    TaskType.VERIFY_COMB: 0.5,
    TaskType.PICK_FIN: 8.0,
    TaskType.INSTALL_FIN: 10.0,
    TaskType.INSPECT_FINS: 10.0,
    TaskType.REVIEW_FINS_CLOSEUP: 6.0,
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


# Backwards-compatible alias: external callers and historical experiment
# snapshots still import ``DEFAULT_DURATIONS``.
DEFAULT_DURATIONS = LEGACY_DURATIONS


def _task_id(order_id: str, unit_index: int, suffix: str) -> str:
    return f"{order_id}_U{unit_index + 1:02d}_{suffix}"


class ProcessPlanTaskGraphBuilder:
    """Generate one graph for every unit plus a shared furnace-batch gate.

    ``catalog`` / ``routing`` / ``resources`` are optional.  When supplied, task
    durations come from parametric capability duration models and
    ``eligible_resources`` is derived from capability declarations filtered by a
    :class:`LineExecutionProfile`.  Without them the builder falls back to the
    legacy constant table and its historical single-resource bindings, which
    keeps unit tests that construct plans in isolation working unchanged.
    """

    def __init__(
        self,
        durations: dict[str | TaskType, float] | None = None,
        *,
        flexible_cell: bool = False,
        camera_coordination: bool = False,
        catalog: CapabilityCatalog | None = None,
        routing: RoutingSpec | None = None,
        resources: Iterable[Any] = (),
        profile: LineExecutionProfile = UNRESTRICTED_PROFILE,
        track_changeover: bool = False,
        fixture_state: FixtureConfiguration | None = None,
    ) -> None:
        # Step D.  Off by default so every existing caller keeps its exact graph;
        # the runtime turns it on and carries fixture state across orders.
        self.track_changeover = bool(track_changeover)
        self.fixture_state = fixture_state or FixtureConfiguration()
        self._changeover_plans: list[dict[str, Any]] = []
        self.durations = dict(LEGACY_DURATIONS)
        for key, value in (durations or {}).items():
            self.durations[TaskType(key)] = float(value)
        # Explicit overrides win over capability-derived durations so callers can
        # still shorten a demo without editing YAML.
        self._duration_overrides = {TaskType(key) for key in (durations or {})}
        self._sequence = 0
        self.flexible_cell = bool(flexible_cell)
        self.camera_coordination = bool(camera_coordination)
        self.catalog = catalog
        self.routing = routing
        self.profile = profile
        resource_list = list(resources)
        self.binder = (
            CapabilityBinder(catalog, resource_list, profile=profile)
            if catalog is not None and resource_list
            else None
        )
        # Populated per build: task_type -> compiled operation data.
        self._compiled: dict[int, dict[str, CompiledOperation]] = {}
        self._binding_snapshots: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ step A
    def _compile_plan(self, plan: ProcessPlan) -> None:
        """Compile the routing for every unit so durations/params come from data."""

        self._compiled = {}
        if self.catalog is None or self.routing is None:
            return
        compiler = RoutingCompiler(self.catalog)
        for assignment in plan.rack_assignments:
            operations = compiler.compile_unit(self.routing, plan, assignment.unit_index)
            by_key: dict[str, CompiledOperation] = {}
            for operation in operations:
                key = operation.task_type
                if operation.fin_index is not None:
                    key = f"{operation.task_type}#{operation.fin_index}"
                by_key[key] = operation
            self._compiled[assignment.unit_index] = by_key

    def _compiled_operation(
        self,
        unit_index: int,
        task_type: TaskType,
        fin_index: int | None = None,
    ) -> CompiledOperation | None:
        unit = self._compiled.get(unit_index)
        if not unit:
            return None
        if fin_index is not None:
            found = unit.get(f"{task_type.value}#{fin_index}")
            if found is not None:
                return found
        return unit.get(task_type.value)

    def _capability_duration(
        self,
        task_type: TaskType,
        operation: CompiledOperation | None,
    ) -> float | None:
        """Parametric duration for a task, or ``None`` to use the legacy table."""

        if task_type in self._duration_overrides:
            return None
        if operation is not None:
            return operation.nominal_duration
        # Topology / changeover / recovery nodes are not in the routing, but
        # their capability still declares a duration model.
        if self.catalog is None:
            return None
        for capability in self.catalog:
            if capability.task_type == task_type.value and not capability.param_schema:
                return capability.duration_for({})
        return None

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
        unit_index: int | None = None,
        fin_index: int | None = None,
    ) -> ManufacturingTask:
        self._sequence += 1
        resources = list(resources)
        payload = dict(payload or {})
        payload.setdefault("route_strategy", plan.route_strategy.value)
        if route_phase is not None:
            payload.setdefault("route_phase", route_phase)
        if task_type is TaskType.REVIEW_BRAZING_CLOSEUP:
            payload.setdefault("route_branch", "S3B_CLOSEUP")
            payload.setdefault("capability", "VISUAL_INSPECTION_BRAZING")
            payload.setdefault("capability_params", {"path_count": len(plan.brazing_paths)})
            payload.setdefault("non_preemptive", True)
        elif task_type is TaskType.REVIEW_FINS_CLOSEUP:
            payload.setdefault("route_branch", "S3B_CLOSEUP")
            payload.setdefault("capability", "VISUAL_INSPECTION_FINS")
            payload.setdefault("capability_params", {"fin_count": len(plan.fin_targets)})
            payload.setdefault("non_preemptive", True)
        duration: float | None = None

        if unit_index is not None:
            operation = self._compiled_operation(unit_index, task_type, fin_index)
            duration = self._capability_duration(task_type, operation)
            if operation is not None:
                resources, payload = self._apply_capability_binding(
                    operation,
                    resources,
                    payload,
                    task_id=task_id,
                )
                selected = payload.get("selected_alternative")
                if isinstance(selected, dict) and isinstance(selected.get("duration"), (int, float)):
                    duration = float(selected["duration"])
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
            estimated_duration=(self.durations.get(task_type, 1.0) if duration is None else duration),
            priority=plan.order.priority,
            retry_limit=retry_limit,
            payload=payload,
            sequence_index=self._sequence,
        )

    # ------------------------------------------------------------------ step B
    def _apply_capability_binding(
        self,
        operation: CompiledOperation,
        resources: list[str],
        payload: dict[str, Any],
        *,
        task_id: str,
    ) -> tuple[list[str], dict[str, Any]]:
        """Replace a hard-coded resource list with capability-derived candidates.

        The capability name, its OR alternatives and every rejection reason are
        recorded on the payload so the console can explain a dispatch decision
        instead of showing an opaque single-resource binding.
        """

        payload["capability"] = operation.capability
        payload["capability_params"] = dict(operation.params)
        if operation.requires_tool_class:
            payload["required_tool_class"] = operation.requires_tool_class
        if not operation.preemptive:
            payload["non_preemptive"] = True

        if self.binder is None:
            return resources, payload

        binding = self.binder.bind(
            operation.capability,
            operation.params,
            base_duration=operation.nominal_duration,
        )
        alternatives: dict[str, Any] = {}
        choices: list[dict[str, Any]] = []
        if operation.alternatives:
            bound_alternatives = self.binder.bind_alternatives(operation.alternatives)
            for option in operation.alternatives:
                mode = str(option.mode)
                result = bound_alternatives[mode]
                alternatives[mode] = result.as_dict()
                choices.append(
                    {
                        "mode": mode,
                        "capability": option.capability,
                        "cost_hint": float(option.cost_hint),
                        "params": dict(option.params),
                        "nominal_duration": float(option.nominal_duration),
                        "candidates": [item.as_dict() for item in result.candidates],
                        "rejected": [
                            {"resource_id": key, "reason": value}
                            for key, value in result.rejected
                        ],
                    }
                )
            payload["capability_alternatives"] = alternatives

        candidates = list(binding.resource_ids)
        if not candidates and not any(choice["candidates"] for choice in choices):
            # Never silently widen or empty the candidate set: keep the authored
            # binding and record why capability binding produced nothing.
            payload["capability_binding_warning"] = (
                f"能力 {operation.capability} 在当前产线没有可用资源，回退到既有绑定"
            )
            payload["capability_rejected"] = [
                {"resource_id": key, "reason": value} for key, value in binding.rejected
            ]
            self._binding_snapshots.append({"task_id": task_id, "fallback": True, **binding.as_dict()})
            return resources, payload

        payload["capability_candidates"] = [item.as_dict() for item in binding.candidates]
        if binding.rejected:
            payload["capability_rejected"] = [
                {"resource_id": key, "reason": value} for key, value in binding.rejected
            ]
        if operation.alternatives:
            # The explicit alternatives are the dispatch decision space.  Keep
            # the authored primary capability as a choice too when the routing
            # author did not repeat it in the OR list; this makes fallback
            # selection work for both forms of route declaration.
            if not any(choice["capability"] == operation.capability for choice in choices):
                choices.insert(
                    0,
                    {
                        "mode": "PRIMARY",
                        "capability": operation.capability,
                        "cost_hint": 1.0,
                        "params": dict(operation.params),
                        "nominal_duration": float(operation.nominal_duration),
                        "candidates": [item.as_dict() for item in binding.candidates],
                        "rejected": [
                            {"resource_id": key, "reason": value}
                            for key, value in binding.rejected
                        ],
                    },
                )
            payload["capability_choices"] = choices
            # A route can fall back to an alternative when its primary
            # capability is unavailable.  The scheduler still sees every
            # viable branch and makes the final choice after reservation.
            candidates = sorted(
                {
                    str(candidate["resource_id"]).upper()
                    for choice in choices
                    for candidate in choice["candidates"]
                }
            )
            viable = [choice for choice in choices if choice["candidates"]]
            if viable:
                selected = min(
                    viable,
                    key=lambda choice: (
                        float(choice["cost_hint"])
                        * min(float(item["duration"]) for item in choice["candidates"]),
                        float(choice["cost_hint"]),
                        min(float(item["duration"]) for item in choice["candidates"]),
                        str(choice["mode"]),
                    ),
                )
                selected_candidate = min(
                    selected["candidates"],
                    key=lambda item: (float(item["duration"]), str(item["resource_id"])),
                )
                payload["selected_alternative"] = {
                    **selected,
                    "selected_resource": str(selected_candidate["resource_id"]).upper(),
                    "duration": float(selected_candidate["duration"]),
                    "selection_source": "planning_default",
                }
        self._binding_snapshots.append({"task_id": task_id, "fallback": False, **binding.as_dict()})
        return candidates, payload

    @property
    def binding_snapshots(self) -> list[dict[str, Any]]:
        """Per-task capability binding records from the most recent build."""

        return list(self._binding_snapshots)

    # ------------------------------------------------------------------ step D
    def _emit_changeover(
        self,
        graph: TaskGraph,
        plan: ProcessPlan,
        *,
        unit_index: int,
        unit_id: str,
        tray_id: str | None,
        predecessor: str,
    ) -> str:
        """Emit the minimal changeover chain, returning the new predecessor.

        Returns ``predecessor`` unchanged when no physical changeover is needed,
        so an unchanged fixture costs exactly zero tasks and zero seconds.
        """

        if not self.track_changeover:
            return predecessor

        target = required_configuration(plan)
        changeover = plan_changeover(
            self.fixture_state,
            target,
            verify=self._uses_high_reliability_route(plan, unit_index),
        )
        self._changeover_plans.append(
            {
                "order_id": plan.order.order_id,
                "unit_id": unit_id,
                "unit_index": unit_index,
                **changeover.as_dict(),
                "nominal_seconds": changeover.duration(self.durations),
            }
        )
        # Advance the tracked state even when nothing changed, so the program
        # identifier follows the product.
        self.fixture_state = target

        current = predecessor
        for action in changeover.actions:
            task = self._make(
                task_id=_task_id(plan.order.order_id, unit_index, action.suffix),
                task_type=action.task_type,
                plan=plan,
                unit_index=unit_index,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=(current,),
                resources=("CHANGEOVER_GANTRY",),
                zones=("ZONE_TABLE2_CORE", "ZONE_CHANGEOVER_GANTRY"),
                route_phase="CHANGEOVER",
                payload={
                    "module_name": action.module_name,
                    "changeover_slot": action.slot,
                    "changeover_kind": action.kind,
                    "changeover_from": changeover.source.as_dict(),
                    "changeover_to": changeover.target.as_dict(),
                },
            )
            graph.add_task(task)
            current = task.task_id
        return current

    @staticmethod
    def _uses_high_reliability_route(plan: ProcessPlan, unit_index: int) -> bool:
        return plan.route_strategy is RouteStrategy.HIGH_RELIABILITY or (
            plan.route_strategy is RouteStrategy.FIRST_ARTICLE and unit_index == 0
        )

    @property
    def changeover_plans(self) -> list[dict[str, Any]]:
        """Per-unit changeover records from the most recent build."""

        return list(self._changeover_plans)

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
            unit_index=index,
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
            unit_index=index,
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
            unit_index=index,
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
            unit_index=index,
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
            unit_index=index,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(dispense.task_id,),
            resources=("ARM3",),
            zones=("ZONE_TABLE2_CORE",),
            retry_limit=2,
            payload={"path_ids": [path.path_id for path in plan.brazing_paths]},
        )
        graph.add_task(inspect_material)
        # Step D: the fixture changeover needed *before* this unit's comb can be
        # configured is derived from (installed configuration → required
        # configuration).  When the previous unit left the right modules mounted
        # the diff is empty and no gantry motion is emitted at all — that is
        # where family-batching savings physically come from.
        comb_predecessor = self._emit_changeover(
            graph,
            plan,
            unit_index=index,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessor=inspect_material.task_id,
        )
        configure_comb = self._make(
            task_id=prefix("CONFIGURE_COMB"),
            task_type=TaskType.CONFIGURE_COMB,
            plan=plan,
            unit_index=index,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=(comb_predecessor,),
            resources=("FIXTURE",),
            zones=("ZONE_TABLE2_CORE",),
            payload={"comb_module_name": plan.fixture_module.name},
        )
        graph.add_task(configure_comb)

        s3b_camera_review = self.camera_coordination and self._uses_high_reliability_route(plan, index)
        review_brazing = None
        if s3b_camera_review:
            review_brazing = self._make(
                task_id=prefix("REVIEW_BRAZING_CLOSEUP"),
                task_type=TaskType.REVIEW_BRAZING_CLOSEUP,
                plan=plan,
                unit_index=index,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=(configure_comb.task_id,),
                resources=("ARM3",),
                zones=("ZONE_TABLE2_CORE",),
                retry_limit=2,
                payload={"inspection_kind": "MATERIAL_INSPECTION", "camera_view": "closeup"},
            )
            graph.add_task(review_brazing)

        install_ids: list[str] = []
        previous_install: str | None = None
        for target in plan.fin_targets:
            # Picking happens at the independent raw-fin magazine and can
            # overlap the visible comb installation at S3.  Only insertion
            # needs the comb lock.  Prefetching the first fin removes the
            # otherwise idle Arm1 gap after the guides visibly seat.
            predecessors = [prepare_tool.task_id]
            if review_brazing is not None:
                predecessors.append(review_brazing.task_id)
            if previous_install is not None:
                predecessors.append(previous_install)
            pick_fin = self._make(
                task_id=prefix(f"PICK_FIN_{target.index + 1:02d}"),
                task_type=TaskType.PICK_FIN,
                plan=plan,
                unit_index=index,
                fin_index=target.index,
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
                unit_index=index,
                fin_index=target.index,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=(pick_fin.task_id, configure_comb.task_id),
                resources=("ARM1",),
                zones=("ZONE_TABLE2_CORE",),
                tool="parallel_gripper",
                retry_limit=2,
                payload={"fin_id": target.fin_id, "target_position": target.position},
            )
            graph.add_task(install_fin)
            previous_install = install_fin.task_id
            install_ids.append(install_fin.task_id)

        review_fins = None
        inspect_fins_predecessors: Iterable[str] = install_ids
        if s3b_camera_review:
            review_fins = self._make(
                task_id=prefix("REVIEW_FINS_CLOSEUP"),
                task_type=TaskType.REVIEW_FINS_CLOSEUP,
                plan=plan,
                unit_index=index,
                unit_id=unit_id,
                tray_id=tray_id,
                predecessors=install_ids,
                resources=("ARM3",),
                zones=("ZONE_TABLE2_CORE",),
                retry_limit=2,
                payload={"inspection_kind": "PRE_BRAZE_INSPECTION", "camera_view": "closeup"},
            )
            graph.add_task(review_fins)
            inspect_fins_predecessors = (review_fins.task_id,)

        inspect_fins = self._make(
            task_id=prefix("INSPECT_FINS"),
            task_type=TaskType.INSPECT_FINS,
            plan=plan,
            unit_index=index,
            unit_id=unit_id,
            tray_id=tray_id,
            predecessors=inspect_fins_predecessors,
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
            unit_index=index,
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
            unit_index=index,
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
            unit_index=index,
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
            unit_index=index,
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
            unit_index=index,
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
            unit_index=index,
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
            for review_key in ("REVIEW_BRAZING_CLOSEUP", "REVIEW_FINS_CLOSEUP"):
                review = items.get(review_key)
                if review is not None:
                    station_task(
                        review,
                        "S3B_ARM3_INSTALL",
                        "ZONE_S3B_ARM3",
                        "VISION_CLOSEUP",
                        "AT_S3B",
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
            # V1/flexible-cell routes keep their historical S1 alignment
            # verification. V2 sets ``camera_coordination`` and replaces this
            # check with the Arm3 S3B close-up nodes above.
            high_reliability = not self.camera_coordination and self._uses_high_reliability_route(plan, index)
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
            if items.get("REVIEW_BRAZING_CLOSEUP") is not None:
                graph.add_dependency(items["INSPECT_BRAZING"].task_id, items["CONFIGURE_COMB"].task_id)

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

    def build(self, plan: ProcessPlan) -> TaskGraph:
        self._sequence = 0
        self._binding_snapshots = []
        self._changeover_plans = []
        # Step A: resolve the data-driven routing before any node is created, so
        # durations and capability bindings come from the plan's real parameters.
        self._compile_plan(plan)
        graph = TaskGraph()
        lock_tasks = [self._build_unit(graph, plan, assignment) for assignment in plan.rack_assignments]
        if self.flexible_cell:
            self._decorate_async_line(graph, plan)
        batch_id = plan.order.order_id
        furnace_by_unit: dict[int, ManufacturingTask] = {}
        single_unit = len(plan.rack_assignments) == 1
        for assignment, lock_task_id in zip(plan.rack_assignments, lock_tasks, strict=True):
            unit_id = f"{batch_id}_UNIT_{assignment.unit_index + 1:02d}"
            prefix = f"{batch_id}" if single_unit else f"{batch_id}_U{assignment.unit_index + 1:02d}"
            batch_ready = self._make(
                task_id=f"{prefix}_BATCH_READY",
                task_type=TaskType.BATCH_READY,
                plan=plan,
                unit_index=assignment.unit_index,
                unit_id=unit_id,
                tray_id=assignment.tray_id,
                predecessors=(lock_task_id,),
                resources=("FURNACE",),
                zones=("ZONE_FURNACE_LOADING",),
                payload={"batch_units": 1},
            )
            graph.add_task(batch_ready)
            furnace = self._make(
                task_id=f"{prefix}_RUN_FURNACE",
                task_type=TaskType.RUN_FURNACE,
                plan=plan,
                unit_index=assignment.unit_index,
                unit_id=unit_id,
                tray_id=assignment.tray_id,
                predecessors=(batch_ready.task_id,),
                resources=("FURNACE",),
                zones=("ZONE_FURNACE_LOADING",),
                payload={
                    "recipe": plan.recipe.name,
                    "duration_s": plan.recipe.to_domain().process_seconds,
                    "batch_units": 1,
                },
            )
            graph.add_task(furnace)
            furnace_by_unit[assignment.unit_index] = furnace

        post_inspection_zones = (
            ("ZONE_OUTFEED", "ZONE_S3_OUTPUT_INTERARM") if self.flexible_cell else ("ZONE_OUTFEED",)
        )
        for assignment in sorted(plan.rack_assignments, key=lambda item: item.layer_index, reverse=True):
            unit_id = f"{plan.order.order_id}_UNIT_{assignment.unit_index + 1:02d}"
            furnace = furnace_by_unit[assignment.unit_index]
            unload = self._make(
                task_id=_task_id(plan.order.order_id, assignment.unit_index, "UNLOAD_RACK"),
                task_type=TaskType.UNLOAD_RACK_LAYER,
                plan=plan,
                unit_index=assignment.unit_index,
                unit_id=unit_id,
                tray_id=assignment.tray_id,
                predecessors=(furnace.task_id,),
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
                unit_index=assignment.unit_index,
                unit_id=unit_id,
                tray_id=assignment.tray_id,
                predecessors=(unload.task_id,),
                resources=("ARM3",),
                zones=post_inspection_zones,
            )
            graph.add_task(inspect)
            route_predecessor = inspect.task_id
            high_reliability = not self.camera_coordination and self._uses_high_reliability_route(
                plan, assignment.unit_index
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
                    unit_index=assignment.unit_index,
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
                unit_index=assignment.unit_index,
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
                unit_index=assignment.unit_index,
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
                    unit_index=assignment.unit_index,
                    unit_id=unit_id,
                    tray_id=assignment.tray_id,
                    predecessors=(route_predecessor,),
                    resources=("OUTFEED",),
                    zones=("ZONE_OUTFEED",),
                    payload={"condition": disposition},
                )
                graph.add_task(route)
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
    capabilities: bool = True,
    resources: Iterable[Any] = (),
    profile: LineExecutionProfile | None = None,
) -> TaskGraph:
    """Build a plan's DAG.

    ``capabilities`` (default) sources durations from the capability ontology.
    Passing ``resources`` additionally enables step-B delayed binding, where
    ``eligible_resources`` is derived from capability declarations filtered by
    ``profile``.  Set ``capabilities=False`` for the fully offline legacy path.
    """

    catalog = default_capability_catalog() if capabilities else None
    routing = default_routing() if capabilities else None
    return ProcessPlanTaskGraphBuilder(
        durations,
        flexible_cell=flexible_cell,
        catalog=catalog,
        routing=routing,
        resources=resources,
        profile=profile or UNRESTRICTED_PROFILE,
    ).build(plan)


__all__ = [
    "DEFAULT_CAPABILITIES_PATH",
    "DEFAULT_DURATIONS",
    "DEFAULT_ROUTING_PATH",
    "LEGACY_DURATIONS",
    "ProcessPlanTaskGraphBuilder",
    "build_task_graph",
    "default_capability_catalog",
    "default_routing",
]
