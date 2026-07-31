"""Step D tests: changeover configuration, setup matrices and KPIs.

The properties that matter:

*   **Zero-cost repeats** — an unchanged fixture must produce no actions at all.
    That is where family-batching savings physically come from, so if this ever
    regresses the whole KPI becomes meaningless.
*   **Sequence dependence** — the cost of a unit depends on what ran before it
    (FJSP-SDST), and grouping same-family orders must measurably beat arrival
    order.
*   **Honest baselines** — the reduction ratio must be computed on one time base
    and must always disclose whether its denominator is measured plant data.
*   **Opt-in** — with tracking off, graphs stay byte-for-byte as before.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from brazing_sim.changeover import (
    PLACEHOLDER_TEACHING_BASELINE,
    ChangeoverKpi,
    FixtureConfiguration,
    TeachingBaseline,
    build_setup_matrix,
    changeover_seconds_from_graph,
    collect_changeover_kpi,
    compare_changeover_baselines,
    configuration_family,
    is_changeover_task,
    plan_changeover,
    required_configuration,
)
from brazing_sim.experiments.metrics_collector import MetricsCollector
from brazing_sim.flexible import build_preset_plan
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.planning import (
    ProcessPlanTaskGraphBuilder,
    default_capability_catalog,
    default_routing,
)
from brazing_sim.planning.task_graph_builder import LEGACY_DURATIONS
from brazing_sim.planning.task_models import TaskType

PRESETS = ("A", "B", "C")


@pytest.fixture(scope="module")
def configurations():
    return {preset: required_configuration(build_preset_plan(preset, quantity=1)) for preset in PRESETS}


def _renamed(preset: str, order_id: str):
    base = build_preset_plan(preset, quantity=1)
    return replace(base, order=replace(base.order, order_id=order_id))


# ------------------------------------------------------------- configuration


def test_each_product_needs_its_own_fixture(configurations):
    """A/B/C use 20/30/15 mm combs, so their signatures must all differ."""

    signatures = {preset: item.signature() for preset, item in configurations.items()}
    assert len(set(signatures.values())) == 3
    assert "20mm" in signatures["A"] and "30mm" in signatures["B"] and "15mm" in signatures["C"]


def test_program_changes_with_product_even_when_fixture_matches():
    """A data-only switch is not a physical changeover."""

    first = FixtureConfiguration(mold="m", comb="c", press="p", program="P1:r")
    second = FixtureConfiguration(mold="m", comb="c", press="p", program="P2:r")
    assert first.matches(second)
    result = plan_changeover(first, second)
    assert result.is_empty
    assert result.program_only is True


def test_identical_configuration_costs_nothing(configurations):
    """The family-batching win, stated as an assertion."""

    result = plan_changeover(configurations["A"], configurations["A"])
    assert result.actions == ()
    assert result.duration(LEGACY_DURATIONS) == pytest.approx(0.0)


def test_cold_line_installs_without_removing(configurations):
    result = plan_changeover(FixtureConfiguration(), configurations["A"])
    kinds = {item.kind for item in result.actions}
    assert "REMOVE" not in kinds
    assert kinds == {"FETCH", "INSTALL", "VERIFY"}


def test_changeover_removes_top_down_and_installs_bottom_up(configurations):
    """A comb cannot be lifted out from under a fitted press."""

    actions = plan_changeover(configurations["A"], configurations["B"]).actions
    removals = [item.slot for item in actions if item.kind == "REMOVE"]
    installs = [item.slot for item in actions if item.kind == "INSTALL"]
    assert removals == ["press", "comb", "mold"]
    assert installs == ["mold", "comb", "press"]
    # Every removal precedes every installation.
    assert max(i for i, a in enumerate(actions) if a.kind == "REMOVE") < min(
        i for i, a in enumerate(actions) if a.kind == "INSTALL"
    )


def test_partial_change_leaves_lower_modules_alone():
    """Changing only the press must not disturb the mold."""

    source = FixtureConfiguration(mold="m1", comb="c1", press="p1")
    target = FixtureConfiguration(mold="m1", comb="c1", press="p2")
    actions = plan_changeover(source, target).actions
    assert {item.slot for item in actions} == {"press"}


def test_configuration_family_counts(configurations):
    counts = configuration_family([configurations[p] for p in "AABC"])
    assert counts[configurations["A"].signature()] == 2
    assert counts[configurations["B"].signature()] == 1


# ---------------------------------------------------------------- setup matrix


def test_setup_matrix_is_sequence_dependent(configurations):
    matrix = build_setup_matrix(configurations.values(), LEGACY_DURATIONS)
    same = matrix.setup_time(configurations["A"], configurations["A"])
    switch = matrix.setup_time(configurations["A"], configurations["B"])
    cold = matrix.setup_time(None, configurations["A"])
    assert same == pytest.approx(0.0)
    assert switch > cold > 0.0  # a swap costs more than a bare install


def test_family_batching_beats_arrival_order(configurations):
    matrix = build_setup_matrix(configurations.values(), LEGACY_DURATIONS)
    queue = [configurations[p] for p in "ABABCC"]
    arrival = matrix.sequence_cost(queue)
    grouped, order = matrix.best_sequence_cost(queue)
    assert grouped < arrival
    # Same multiset of orders, only resequenced.
    assert sorted(item.signature() for item in order) == sorted(item.signature() for item in queue)
    # Grouped runs contain no repeated family blocks.
    signatures = [item.signature() for item in order]
    assert len(set(signatures)) == len({*signatures})
    blocks = [key for index, key in enumerate(signatures) if index == 0 or signatures[index - 1] != key]
    assert len(blocks) == len(set(blocks)), "同族订单应连续排列"


def test_setup_matrix_snapshot_is_serialisable(configurations):
    matrix = build_setup_matrix(configurations.values(), LEGACY_DURATIONS)
    snapshot = matrix.as_dict()
    assert snapshot["entries"] and "signatures" in snapshot
    assert all({"from", "to", "seconds"} <= set(item) for item in snapshot["entries"])


# --------------------------------------------------------------- graph wiring


def test_changeover_is_opt_in_and_default_graph_is_unchanged():
    """Existing callers must get byte-for-byte the same graph."""

    plan = build_preset_plan("A", quantity=3)
    off = ProcessPlanTaskGraphBuilder(flexible_cell=True).build(plan)
    on = ProcessPlanTaskGraphBuilder(
        flexible_cell=True,
        catalog=default_capability_catalog(),
        routing=default_routing(),
        track_changeover=True,
    ).build(plan)
    assert len(on) > len(off)
    assert changeover_seconds_from_graph(off)[1] == 0


def test_repeated_units_of_one_order_pay_setup_once():
    """Three A units share a fixture, so only the first pays."""

    builder = ProcessPlanTaskGraphBuilder(
        flexible_cell=True,
        catalog=default_capability_catalog(),
        routing=default_routing(),
        track_changeover=True,
    )
    builder.build(build_preset_plan("A", quantity=3))
    records = builder.changeover_plans
    assert len(records) == 3
    assert records[0]["action_count"] > 0
    assert [item["action_count"] for item in records[1:]] == [0, 0]


def test_builder_carries_fixture_state_between_orders(configurations):
    builder = ProcessPlanTaskGraphBuilder(
        flexible_cell=True,
        catalog=default_capability_catalog(),
        routing=default_routing(),
        track_changeover=True,
    )
    builder.build(_renamed("A", "FIRST_A"))
    assert builder.fixture_state.signature() == configurations["A"].signature()
    builder.build(_renamed("A", "SECOND_A"))
    assert builder.changeover_plans[-1]["action_count"] == 0
    builder.build(_renamed("B", "THEN_B"))
    assert builder.changeover_plans[-1]["action_count"] > 0


def test_changeover_tasks_are_distinguishable_from_post_braze_teardown():
    """``REMOVE_OLD_COMB`` serves both setup and teardown; only one is setup."""

    builder = ProcessPlanTaskGraphBuilder(
        flexible_cell=True,
        catalog=default_capability_catalog(),
        routing=default_routing(),
        track_changeover=True,
    )
    graph = builder.build(build_preset_plan("A", quantity=1))
    teardown = [
        task
        for task in graph
        if task.task_type is TaskType.REMOVE_OLD_COMB and task.payload.get("after_brazing")
    ]
    assert teardown, "焊后拆解任务应存在"
    assert all(not is_changeover_task(task) for task in teardown)
    seconds, count = changeover_seconds_from_graph(graph)
    assert count == builder.changeover_plans[0]["action_count"]
    assert seconds == pytest.approx(builder.changeover_plans[0]["nominal_seconds"])


# -------------------------------------------------------------------- runtime


def test_runtime_tracks_fixture_across_orders():
    runtime = ManufacturingRuntime(flexible_cell=True, track_changeover=True)
    for index, preset in enumerate("ABAB"):
        runtime.submit_plan(_renamed(preset, f"O{index}_{preset}"), now=0.0)
    assert len(runtime.changeover_log) == 4
    assert sum(item["action_count"] for item in runtime.changeover_log) > 0
    runtime.reset(0.0)
    assert runtime.changeover_log == []
    assert runtime.installed_fixture.is_empty


def test_runtime_sequence_order_changes_total_setup_time():
    def total(sequence):
        runtime = ManufacturingRuntime(flexible_cell=True, track_changeover=True)
        for index, preset in enumerate(sequence):
            runtime.submit_plan(_renamed(preset, f"O{index}_{preset}"), now=0.0)
        return sum(item["nominal_seconds"] for item in runtime.changeover_log)

    arrival = total("ABABCC")
    grouped = total("AABBCC")
    assert grouped < arrival
    assert (arrival - grouped) / arrival > 0.2


def test_runtime_publishes_setup_cost_to_the_scheduler():
    """``product_changeover_cost`` was a dead weight before step D."""

    runtime = ManufacturingRuntime(flexible_cell=True, track_changeover=True)
    runtime.submit_plan(_renamed("A", "COST_A"), now=0.0)
    ready = runtime._refresh_ready(0.0)
    runtime._annotate_changeover_cost(ready)
    annotated = [task for task in ready if "product_changeover_cost" in task.payload]
    assert annotated, "就绪任务应带上换型成本"


def test_changeover_tracking_is_off_by_default():
    runtime = ManufacturingRuntime(flexible_cell=True)
    runtime.submit_plan(build_preset_plan("A", quantity=1), now=0.0)
    assert runtime.changeover_log == []


# ------------------------------------------------------------------------ KPI


def test_metrics_collector_reports_the_three_headline_numbers():
    runtime = ManufacturingRuntime(flexible_cell=True, track_changeover=True)
    for index, preset in enumerate("ABAB"):
        runtime.submit_plan(_renamed(preset, f"O{index}_{preset}"), now=0.0)
    metrics = MetricsCollector().calculate(runtime, now=100.0)
    for key in (
        "changeover_seconds",
        "changeover_count",
        "changeover_ratio_vs_baseline",
    ):
        assert key in metrics
    assert 0.0 <= metrics["changeover_ratio_vs_baseline"] <= 1.0


def test_runtime_log_and_graph_agree_on_changeover_time():
    """Two independent measurements; a mismatch means one is counting wrong."""

    runtime = ManufacturingRuntime(flexible_cell=True, track_changeover=True)
    for index, preset in enumerate("ABC"):
        runtime.submit_plan(_renamed(preset, f"O{index}_{preset}"), now=0.0)
    metrics = MetricsCollector().calculate(runtime, now=100.0)
    assert metrics["changeover_seconds"] == pytest.approx(metrics["changeover_seconds_from_graph"])
    assert metrics["changeover_action_count"] == metrics["changeover_task_count"]


def test_kpi_only_counts_effective_changeovers():
    log = [
        {"action_count": 7, "nominal_seconds": 16.0},
        {"action_count": 0, "nominal_seconds": 0.0},
        {"action_count": 0, "nominal_seconds": 0.0},
    ]
    kpi = collect_changeover_kpi(log, PLACEHOLDER_TEACHING_BASELINE)
    assert kpi.changeover_count == 1  # two units needed nothing
    assert kpi.changeover_action_count == 7
    assert kpi.changeover_seconds == pytest.approx(16.0)


def test_ratio_uses_one_time_base():
    """A real-seconds baseline must be demo-scaled before dividing.

    Without ``demo_scale`` a 30-minute manual window over a 16-second automatic
    swap reports ~99%, which measures the demo compression rather than the
    automation.
    """

    unscaled = TeachingBaseline(1800.0, source="test", measured=True, demo_scale=1.0)
    scaled = TeachingBaseline(1800.0, source="test", measured=True, demo_scale=1.0 / 15.0)
    assert scaled.demo_seconds_per_changeover == pytest.approx(120.0)
    log = [{"action_count": 7, "nominal_seconds": 16.0}]
    assert collect_changeover_kpi(log, unscaled).changeover_ratio_vs_baseline > 0.99
    honest = collect_changeover_kpi(log, scaled).changeover_ratio_vs_baseline
    assert 0.7 < honest < 0.95


def test_kpi_discloses_a_placeholder_baseline():
    """A guess must never be presentable as plant data."""

    kpi = collect_changeover_kpi(
        [{"action_count": 7, "nominal_seconds": 16.0}],
        PLACEHOLDER_TEACHING_BASELINE,
    )
    assert kpi.as_dict()["baseline_is_placeholder"] is True
    assert PLACEHOLDER_TEACHING_BASELINE.measured is False
    assert "现场" in PLACEHOLDER_TEACHING_BASELINE.notes_zh


def test_teaching_baseline_requires_a_source():
    with pytest.raises(ValueError):
        TeachingBaseline(1800.0, source="   ")
    with pytest.raises(ValueError):
        TeachingBaseline(-1.0, source="test")
    with pytest.raises(ValueError):
        TeachingBaseline(1800.0, source="test", demo_scale=0.0)


def test_three_tier_comparison_separates_automation_from_sequencing(configurations):
    result = compare_changeover_baselines(
        [configurations[p] for p in "ABABCC"],
        LEGACY_DURATIONS,
        PLACEHOLDER_TEACHING_BASELINE,
    )
    names = [tier["name"] for tier in result["tiers"]]
    assert names == ["MANUAL_TEACHING", "AUTOMATIC_UNSORTED", "AUTOMATIC_FAMILY_BATCHED"]
    seconds = [tier["changeover_seconds"] for tier in result["tiers"]]
    assert seconds[0] > seconds[1] > seconds[2]  # each tier is an improvement

    improvements = result["improvements"]
    assert improvements["automation_and_sequencing_ratio"] > improvements["automation_ratio"]
    # Sequencing-only is a ratio of two simulated numbers, so it is independent
    # of the manual baseline and of the demo time base — the robust headline.
    assert improvements["sequencing_only_ratio"] > 0.2
    assert improvements["sequencing_saved_seconds"] > 0.0


def test_three_tier_comparison_handles_an_empty_queue():
    result = compare_changeover_baselines([], LEGACY_DURATIONS, PLACEHOLDER_TEACHING_BASELINE)
    assert result["unit_count"] == 0
    assert result["tiers"] == []


def test_kpi_ratio_is_clamped():
    slow = TeachingBaseline(1.0, source="test", measured=True)
    kpi = collect_changeover_kpi([{"action_count": 9, "nominal_seconds": 999.0}], slow)
    assert kpi.changeover_ratio_vs_baseline == 0.0  # never negative


def test_changeover_share_of_occupied_time():
    kpi = ChangeoverKpi(
        changeover_seconds=25.0,
        changeover_count=1,
        changeover_action_count=7,
        baseline_seconds=120.0,
        baseline_source="test",
        baseline_measured=True,
        productive_seconds=75.0,
    )
    assert kpi.changeover_share == pytest.approx(0.25)
