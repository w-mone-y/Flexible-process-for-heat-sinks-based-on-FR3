"""Step A/B tests: data-driven process routing and capability delayed binding.

Two properties matter most and are asserted here:

*   **Equivalence** — compiling the shipped routing must reproduce the graph the
    hand-written builder produced (same nodes, same edges, same durations), so
    the refactor cannot silently change V1/V2 behaviour.
*   **Flexibility** — a capability with several declaring resources must yield
    several candidates, filtered by tool class, parameter window and the line's
    execution profile, with a Chinese reason for every rejection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brazing_sim.flexible import (
    DurationModel,
    DurationModelError,
    FlexibleConfigError,
    build_preset_plan,
    compile_routing,
    load_capabilities,
    load_routing,
    parse_resource_capabilities,
    plan_parameter_bindings,
)
from brazing_sim.flexible.routing_compiler import RoutingCompileError
from brazing_sim.manufacturing_config import load_resource_config
from brazing_sim.paths import CONFIG_DIR
from brazing_sim.planning import (
    V1_SHALLOW_U_PROFILE,
    V2_DUAL_INSTALL_PROFILE,
    CapabilityBinder,
    ProcessPlanTaskGraphBuilder,
    default_capability_catalog,
    default_routing,
)
from brazing_sim.planning.task_models import TaskType

CAPABILITIES_PATH = CONFIG_DIR / "capabilities.yaml"
ROUTING_PATH = CONFIG_DIR / "routings" / "heat_sink_standard.yaml"


@pytest.fixture(scope="module")
def catalog():
    return default_capability_catalog()


@pytest.fixture(scope="module")
def routing():
    return default_routing()


@pytest.fixture(scope="module")
def resources():
    states, _zones = load_resource_config(CONFIG_DIR / "resources.yaml")
    return states


# --------------------------------------------------------------- duration model


def test_duration_model_evaluates_parametric_expression():
    model = DurationModel("2.4 + 0.9 * path_count", allowed_names=frozenset({"path_count"}))
    assert model.evaluate({"path_count": 10}) == pytest.approx(11.4)
    assert model.parameter_names == frozenset({"path_count"})


def test_duration_model_supports_envelope_helpers():
    model = DurationModel("max(24.0, 1.5 * path_count)", allowed_names=frozenset({"path_count"}))
    assert model.evaluate({"path_count": 4}) == pytest.approx(24.0)
    assert model.evaluate({"path_count": 40}) == pytest.approx(60.0)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "open('/etc/passwd').read()",
        "path_count if path_count else 0",
        "[1, 2, 3]",
        "lambda: 1",
    ],
)
def test_duration_model_rejects_non_arithmetic_expressions(expression):
    with pytest.raises(DurationModelError):
        DurationModel(expression, allowed_names=frozenset({"path_count"}))


def test_duration_model_rejects_undeclared_parameter():
    with pytest.raises(DurationModelError) as excinfo:
        DurationModel("2.0 * fin_count", allowed_names=frozenset({"path_count"}))
    assert "fin_count" in str(excinfo.value)


def test_duration_model_rejects_division_by_zero_and_negative_result():
    model = DurationModel("10.0 / value", allowed_names=frozenset({"value"}))
    with pytest.raises(DurationModelError):
        model.evaluate({"value": 0})
    negative = DurationModel("0.0 - value", allowed_names=frozenset({"value"}))
    with pytest.raises(DurationModelError):
        negative.evaluate({"value": 5})


# ------------------------------------------------------------------- catalogue


def test_shipped_capability_catalog_covers_every_task_type(catalog):
    """No TaskType may lack a duration model, or the constant table cannot retire."""

    covered = {capability.task_type for capability in catalog}
    missing = sorted(item.value for item in TaskType if item.value not in covered)
    assert missing == []


def test_capability_rejects_unknown_and_out_of_range_parameters(catalog):
    capability = catalog.get("MATERIAL_DISPENSING_DUAL")
    _, unknown = capability.normalize_params({"path_count": 10, "speed_m_s": 0.1, "typo_field": 1})
    assert "typo_field" in unknown
    _, out_of_range = capability.normalize_params({"path_count": 999, "speed_m_s": 0.1})
    assert "path_count" in out_of_range


def test_capability_fills_declared_defaults(catalog):
    capability = catalog.get("MATERIAL_DISPENSING_DUAL")
    resolved, error = capability.normalize_params({"path_count": 10, "speed_m_s": 0.1})
    assert error == ""
    assert resolved["bead_offset_m"] == pytest.approx(0.0025)


def test_unknown_capability_field_is_rejected(tmp_path: Path):
    bad = tmp_path / "capabilities.yaml"
    bad.write_text(
        "schema_version: 1\n"
        "capabilities:\n"
        "  DEMO:\n"
        "    task_type: LOCK_FIXTURE\n"
        "    duration_model: '1.0'\n"
        "    unexpected_field: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(FlexibleConfigError) as excinfo:
        load_capabilities(bad)
    assert "unexpected_field" in str(excinfo.value)


# --------------------------------------------------------------------- routing


def test_routing_alternatives_must_share_process_effects(tmp_path: Path, catalog):
    """An OR branch that does not produce the same effect is not a substitute."""

    bad = tmp_path / "routing.yaml"
    bad.write_text(
        "schema_version: 1\n"
        "routing_id: BAD\n"
        "product: DEMO\n"
        "operations:\n"
        "  - id: OP10\n"
        "    capability: BASE_PICKING\n"
        "  - id: OP20\n"
        "    capability: BASE_LOADING\n"
        "    after: [OP10]\n"
        "    alternatives:\n"
        "      - {mode: NORMAL, capability: BASE_LOADING}\n"
        "      - {mode: WRONG, capability: FIXTURE_LOCKING}\n",
        encoding="utf-8",
    )
    with pytest.raises(FlexibleConfigError) as excinfo:
        load_routing(bad, catalog=catalog)
    assert "工艺效果" in str(excinfo.value)


def test_routing_rejects_forward_reference(tmp_path: Path, catalog):
    bad = tmp_path / "routing.yaml"
    bad.write_text(
        "schema_version: 1\n"
        "routing_id: BAD\n"
        "product: DEMO\n"
        "operations:\n"
        "  - id: OP10\n"
        "    capability: BASE_PICKING\n"
        "    after: [OP99_LATER]\n",
        encoding="utf-8",
    )
    with pytest.raises(FlexibleConfigError):
        load_routing(bad, catalog=catalog)


def test_routing_rejects_unsatisfied_precondition(tmp_path: Path, catalog):
    """``BASE_LOADING`` needs ``base_picked``; without the pick it must fail."""

    bad = tmp_path / "routing.yaml"
    bad.write_text(
        "schema_version: 1\n"
        "routing_id: BAD\n"
        "product: DEMO\n"
        "operations:\n"
        "  - id: OP20\n"
        "    capability: BASE_LOADING\n",
        encoding="utf-8",
    )
    with pytest.raises(FlexibleConfigError) as excinfo:
        load_routing(bad, catalog=catalog)
    assert "前置条件" in str(excinfo.value)


def test_shipped_routing_declares_expected_or_branches(routing):
    dispense = routing.operation("OP20_DISPENSE")
    inspect = routing.operation("OP25_INSPECT_BRAZING")
    assert {item.mode for item in dispense.alternatives} == {"DUAL_NOZZLE", "SINGLE_TWO_PASS"}
    assert {item.mode for item in inspect.alternatives} == {"ARM_HANDHELD", "FIXED_GANTRY"}
    assert dispense.has_alternatives and inspect.has_alternatives


# -------------------------------------------------------------------- compiler


@pytest.mark.parametrize("preset,fins", [("A", 5), ("B", 4), ("C", 7)])
def test_compiler_expands_one_node_per_fin(preset, fins, catalog, routing):
    plan = build_preset_plan(preset, quantity=1)
    operations = compile_routing(routing, plan, catalog)[0]
    installs = [item for item in operations if item.operation_id == "OP35_INSTALL_FIN"]
    assert len(installs) == fins
    assert [item.fin_index for item in installs] == list(range(fins))


def test_compiler_serialises_fins_through_after_previous(catalog, routing):
    """One gripper holds one fin: pick *i* waits on install *i-1*."""

    plan = build_preset_plan("A", quantity=1)
    operations = {item.node_id: item for item in compile_routing(routing, plan, catalog)[0]}
    first_pick = operations["U01_OP30_PICK_FIN_F01"]
    second_pick = operations["U01_OP30_PICK_FIN_F02"]
    assert "U01_OP35_INSTALL_FIN_F01" not in first_pick.predecessors
    assert "U01_OP35_INSTALL_FIN_F01" in second_pick.predecessors


def test_compiler_resolves_plan_placeholders(catalog, routing):
    plan = build_preset_plan("C", quantity=1)
    bindings = plan_parameter_bindings(plan)
    operations = {item.operation_id: item for item in compile_routing(routing, plan, catalog)[0]}
    dispense = operations["OP20_DISPENSE"]
    assert dispense.params["path_count"] == bindings["path_count"] == len(plan.brazing_paths)
    press = operations["OP45_APPLY_PRESS"]
    assert press.params["target_force_n"] == pytest.approx(plan.product.target_clamping_force_n)


def test_preset_durations_stay_inside_the_compatibility_envelope(catalog, routing):
    """A/B/C deliberately keep their pre-refactor tempo.

    The shipped duration models carry a ``max(envelope, ...)`` floor so the three
    existing presets reproduce the constant table exactly (see the equivalence
    test below).  Growth therefore only shows up beyond the envelope, which is
    what :func:`test_duration_model_scales_beyond_the_envelope` covers.
    """

    def inspect_duration(preset):
        plan = build_preset_plan(preset, quantity=1)
        operations = {item.operation_id: item for item in compile_routing(routing, plan, catalog)[0]}
        return operations["OP40_INSPECT_FINS"].nominal_duration

    assert inspect_duration("B") == pytest.approx(10.0)
    assert inspect_duration("C") == pytest.approx(10.0)


def test_duration_model_scales_beyond_the_envelope(catalog):
    """A larger product must take longer with no code change at all."""

    inspection = catalog.get("VISUAL_INSPECTION_FINS")
    assert inspection.duration_for({"fin_count": 7}) == pytest.approx(10.0)
    assert inspection.duration_for({"fin_count": 12}) == pytest.approx(13.6)

    dispensing = catalog.get("MATERIAL_DISPENSING_DUAL")
    base = {"speed_m_s": 0.1, "bead_offset_m": 0.0025}
    assert dispensing.duration_for({**base, "path_count": 14}) == pytest.approx(24.0)
    assert dispensing.duration_for({**base, "path_count": 24}) == pytest.approx(38.4)


def test_compiler_reports_unknown_placeholder(catalog, tmp_path: Path):
    bad = tmp_path / "routing.yaml"
    bad.write_text(
        "schema_version: 1\n"
        "routing_id: BAD\n"
        "product: DEMO\n"
        "operations:\n"
        "  - id: OP10\n"
        "    capability: BASE_PICKING\n"
        "  - id: OP20\n"
        "    capability: MATERIAL_DISPENSING_DUAL\n"
        "    after: [OP10]\n"
        "    params:\n"
        "      path_count: $not_a_plan_field\n"
        "      speed_m_s: 0.1\n",
        encoding="utf-8",
    )
    # BASE_LOADING is missing, so precondition validation must be skipped here.
    spec = load_routing(bad)
    plan = build_preset_plan("A", quantity=1)
    with pytest.raises(RoutingCompileError) as excinfo:
        compile_routing(spec, plan, catalog)
    assert "not_a_plan_field" in str(excinfo.value)


# ------------------------------------------------------- equivalence guarantee


@pytest.mark.parametrize("preset", ["A", "B", "C"])
@pytest.mark.parametrize("quantity", [1, 3])
@pytest.mark.parametrize("flexible_cell", [False, True])
def test_capability_graph_matches_legacy_graph(preset, quantity, flexible_cell, catalog, routing):
    """The data-driven build must reproduce the hand-written DAG exactly."""

    plan = build_preset_plan(preset, quantity=quantity)
    legacy = ProcessPlanTaskGraphBuilder(flexible_cell=flexible_cell).build(plan)
    compiled = ProcessPlanTaskGraphBuilder(
        flexible_cell=flexible_cell,
        catalog=catalog,
        routing=routing,
    ).build(plan)

    assert set(legacy.tasks) == set(compiled.tasks)
    for task_id in sorted(legacy.tasks):
        old, new = legacy.get(task_id), compiled.get(task_id)
        assert sorted(old.predecessors) == sorted(new.predecessors), task_id
        assert sorted(old.successors) == sorted(new.successors), task_id
        assert old.eligible_resources == new.eligible_resources, task_id
        assert old.required_zones == new.required_zones, task_id
        assert old.estimated_duration == pytest.approx(new.estimated_duration), task_id


def test_explicit_duration_override_still_wins(catalog, routing):
    plan = build_preset_plan("A", quantity=1)
    graph = ProcessPlanTaskGraphBuilder(
        {TaskType.DISPENSE_BRAZING: 3.0},
        catalog=catalog,
        routing=routing,
    ).build(plan)
    dispense = next(task for task in graph if task.task_type is TaskType.DISPENSE_BRAZING)
    assert dispense.estimated_duration == pytest.approx(3.0)


# ------------------------------------------------------------ delayed binding


def test_resource_config_exposes_process_capabilities(resources):
    by_id = {item.resource_id: item for item in resources}
    assert by_id["ARM3"].has_capability("FIN_ASSEMBLY")
    assert by_id["ARM1"].has_capability("FIN_ASSEMBLY")
    assert by_id["ARM3"].speed_factor_for("FIN_ASSEMBLY") == pytest.approx(1.35)
    assert by_id["ARM1"].tools_for_class("GRIPPER") == ("parallel_gripper",)


def test_binder_offers_both_arms_and_prefers_the_faster_one(catalog, resources):
    binder = CapabilityBinder(catalog, resources, profile=V2_DUAL_INSTALL_PROFILE)
    result = binder.bind("FIN_ASSEMBLY", {"fin_pitch_m": 0.020})
    assert result.resource_ids == ("ARM3", "ARM1")
    assert result.candidates[0].duration < result.candidates[1].duration


def test_binder_enforces_resource_parameter_window(catalog, resources):
    """Arm3's narrow gripper cannot reach a 40 mm pitch, and says why."""

    binder = CapabilityBinder(catalog, resources, profile=V2_DUAL_INSTALL_PROFILE)
    result = binder.bind("FIN_ASSEMBLY", {"fin_pitch_m": 0.040})
    assert result.resource_ids == ("ARM1",)
    reasons = dict(result.rejected)
    assert "ARM3" in reasons
    assert "fin_pitch_m" in reasons["ARM3"]


def test_line_profile_restricts_v1_fin_assembly_to_arm1(catalog, resources):
    """V1's fin skills weld to arm1, so Arm3 must not be offered there."""

    v1 = CapabilityBinder(catalog, resources, profile=V1_SHALLOW_U_PROFILE)
    v2 = CapabilityBinder(catalog, resources, profile=V2_DUAL_INSTALL_PROFILE)
    params = {"fin_pitch_m": 0.020}
    assert v1.bind("FIN_ASSEMBLY", params).resource_ids == ("ARM1",)
    assert set(v2.bind("FIN_ASSEMBLY", params).resource_ids) == {"ARM1", "ARM3"}
    reasons = dict(v1.bind("FIN_ASSEMBLY", params).rejected)
    assert "ARM3" in reasons and V1_SHALLOW_U_PROFILE.name in reasons["ARM3"]


def test_binder_requires_a_tool_of_the_declared_class(catalog, resources):
    """A dispenser capability must not bind to an arm with no dispenser."""

    binder = CapabilityBinder(catalog, resources, profile=V2_DUAL_INSTALL_PROFILE)
    result = binder.bind("MATERIAL_DISPENSING_DUAL", {"path_count": 10, "speed_m_s": 0.1})
    assert result.resource_ids == ("ARM2",)


def test_binder_reports_alternatives_per_mode(catalog, routing, resources):
    plan = build_preset_plan("A", quantity=1)
    operations = {item.operation_id: item for item in compile_routing(routing, plan, catalog)[0]}
    binder = CapabilityBinder(catalog, resources, profile=V2_DUAL_INSTALL_PROFILE)
    bound = binder.bind_alternatives(operations["OP20_DISPENSE"].alternatives)
    assert set(bound) == {"DUAL_NOZZLE", "SINGLE_TWO_PASS"}
    assert bound["DUAL_NOZZLE"].resource_ids == ("ARM2",)
    # The slower single-nozzle route stays visible as a real fallback.
    assert bound["SINGLE_TWO_PASS"].nominal_duration > bound["DUAL_NOZZLE"].nominal_duration


def test_builder_records_capability_metadata_on_tasks(catalog, routing, resources):
    plan = build_preset_plan("A", quantity=1)
    graph = ProcessPlanTaskGraphBuilder(
        flexible_cell=True,
        catalog=catalog,
        routing=routing,
        resources=resources,
        profile=V2_DUAL_INSTALL_PROFILE,
    ).build(plan)
    install = next(task for task in graph if task.task_type is TaskType.INSTALL_FIN)
    assert install.payload["capability"] == "FIN_ASSEMBLY"
    assert install.payload["non_preemptive"] is True
    assert len(install.eligible_resources) == 2
    dispense = next(task for task in graph if task.task_type is TaskType.DISPENSE_BRAZING)
    assert set(dispense.payload["capability_alternatives"]) == {
        "DUAL_NOZZLE",
        "SINGLE_TWO_PASS",
    }


def test_delayed_binding_creates_a_real_decision_space(catalog, routing, resources):
    """Before step B every node had exactly one candidate; V2 must have many."""

    plan = build_preset_plan("A", quantity=3)
    graph = ProcessPlanTaskGraphBuilder(
        flexible_cell=True,
        catalog=catalog,
        routing=routing,
        resources=resources,
        profile=V2_DUAL_INSTALL_PROFILE,
    ).build(plan)
    flexible = [task for task in graph if len(task.eligible_resources) > 1]
    assert len(flexible) >= plan.quantity * len(plan.fin_targets)


def test_v1_profile_keeps_single_resource_bindings(catalog, routing, resources):
    """Delayed binding must not hand V1 a resource its actors cannot drive."""

    plan = build_preset_plan("A", quantity=1)
    graph = ProcessPlanTaskGraphBuilder(
        flexible_cell=True,
        catalog=catalog,
        routing=routing,
        resources=resources,
        profile=V1_SHALLOW_U_PROFILE,
    ).build(plan)
    for task in graph:
        if task.task_type in {TaskType.PICK_FIN, TaskType.INSTALL_FIN}:
            assert task.eligible_resources == ["ARM1"], task.task_id


def test_binding_falls_back_when_no_resource_declares_the_capability(catalog, routing):
    """An empty candidate set must never erase the authored binding."""

    states, _zones = load_resource_config(CONFIG_DIR / "resources.yaml")
    without_arm1 = [item for item in states if item.resource_id != "ARM1"]
    plan = build_preset_plan("A", quantity=1)
    graph = ProcessPlanTaskGraphBuilder(
        catalog=catalog,
        routing=routing,
        resources=without_arm1,
        profile=V1_SHALLOW_U_PROFILE,
    ).build(plan)
    install = next(task for task in graph if task.task_type is TaskType.INSTALL_FIN)
    assert install.eligible_resources == ["ARM1"]
    assert "capability_binding_warning" in install.payload


# --------------------------------------------------- resource capability parsing


def test_new_product_needs_no_code_change(catalog, routing, resources):
    """Product D exists only as YAML: the headline claim of step A.

    Adding it required ``config/products/product_d.yaml`` plus
    ``config/orders/order_004.yaml`` and no Python at all.  Its tempo must scale
    past the A/B/C compatibility envelope purely from ``duration_model``.
    """

    from brazing_sim.flexible import build_process_plan

    plan = build_process_plan(CONFIG_DIR / "orders" / "order_004.yaml")
    assert plan.product.preset == "D"
    assert len(plan.fin_targets) == 9
    assert len(plan.brazing_paths) == 18

    graph = ProcessPlanTaskGraphBuilder(
        flexible_cell=True,
        catalog=catalog,
        routing=routing,
        resources=resources,
        profile=V2_DUAL_INSTALL_PROFILE,
    ).build(plan)

    dispense = next(task for task in graph if task.task_type is TaskType.DISPENSE_BRAZING)
    inspect = next(task for task in graph if task.task_type is TaskType.INSPECT_FINS)
    # 18 paths / 9 fins push both operations beyond the A/B/C envelopes.
    assert dispense.estimated_duration == pytest.approx(29.4)
    assert inspect.estimated_duration == pytest.approx(11.2)
    # 15 mm pitch is inside Arm3's window, so both arms can install.
    installs = [task for task in graph if task.task_type is TaskType.INSTALL_FIN]
    assert len(installs) == 9
    assert all(len(task.eligible_resources) == 2 for task in installs)


def test_fixture_capacity_still_rejects_impossible_products(tmp_path: Path):
    """Data-driven flexibility must not weaken physical validation."""

    from brazing_sim.flexible import FlexibleConfigError as ConfigError
    from brazing_sim.flexible import build_process_plan

    product = (CONFIG_DIR / "products" / "product_d.yaml").read_text(encoding="utf-8")
    # 12 fins cannot fit the 9-slot 15 mm comb module.
    (tmp_path / "products").mkdir()
    (tmp_path / "products" / "product_x.yaml").write_text(
        product.replace("fin_count: 9", "fin_count: 12"), encoding="utf-8"
    )
    (tmp_path / "orders").mkdir()
    (tmp_path / "orders" / "order_x.yaml").write_text(
        "schema_version: 1\n"
        "order_id: ORDER_X\n"
        "product: ../products/product_x.yaml\n"
        "quantity: 1\n"
        "priority: 10\n"
        "due_time: null\n"
        "preferred_rack_layer: null\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as excinfo:
        build_process_plan(tmp_path / "orders" / "order_x.yaml")
    assert "梳齿槽数不足" in str(excinfo.value)


def test_parse_resource_capabilities_accepts_legacy_string_form():
    parsed = parse_resource_capabilities(
        ["FIN_ASSEMBLY", {"name": "FIN_PICKING", "speed_factor": 2.0}],
        source=Path("test.yaml"),
        path="resources[0].process_capabilities",
    )
    assert [item.name for item in parsed] == ["FIN_ASSEMBLY", "FIN_PICKING"]
    assert parsed[0].speed_factor == pytest.approx(1.0)
    assert parsed[1].speed_factor == pytest.approx(2.0)


def test_parse_resource_capabilities_rejects_inverted_window():
    with pytest.raises(FlexibleConfigError):
        parse_resource_capabilities(
            [{"name": "FIN_ASSEMBLY", "param_limits": {"fin_pitch_m": [0.030, 0.015]}}],
            source=Path("test.yaml"),
            path="resources[0].process_capabilities",
        )


def test_parse_resource_capabilities_rejects_duplicates():
    with pytest.raises(FlexibleConfigError):
        parse_resource_capabilities(
            ["FIN_ASSEMBLY", "FIN_ASSEMBLY"],
            source=Path("test.yaml"),
            path="resources[0].process_capabilities",
        )
