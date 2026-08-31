"""The flexibility report must be evidence, not marketing.

Two properties are asserted:

*   **Every dimension carries a number.** A dimension with a slogan and no metric
    is unfalsifiable, which is exactly what a judge should distrust.
*   **``PARTIAL`` stays ``PARTIAL``.** Process flexibility models alternative
    routes but no scheduler chooses between them; changeover ratios lean on a
    placeholder baseline.  If either silently became ``FULL`` the report would be
    overstating the work, so those states are pinned.
"""

from __future__ import annotations

import pytest

from brazing_sim.flexibility_report import FULL, NONE, PARTIAL, flexibility_report


def test_committed_task_records_the_selected_alternative_mode() -> None:
    from brazing_sim.manufacturing_runtime import ManufacturingRuntime
    from brazing_sim.planning import ManufacturingTask, TaskType

    task = ManufacturingTask(
        task_id="ALT-1",
        task_type=TaskType.INSTALL_FIN,
        order_id="ALT",
        unit_id="ALT-UNIT",
        payload={
            "capability_alternatives": {
                "ARM1_FAST": {"candidates": [{"resource_id": "ARM1", "duration": 3.0}]},
                "ARM3_SAFE": {"candidates": [{"resource_id": "ARM3", "duration": 4.0}]},
            }
        },
    )

    ManufacturingRuntime._record_alternative_selection(task, "ARM3")

    assert task.payload["selected_alternative"] == "ARM3_SAFE"
    assert "ARM3" in task.payload["alternative_selection_reason"]


@pytest.fixture(scope="module")
def report():
    return flexibility_report({"line_profile": "V2_DUAL_INSTALL"})


def test_all_six_dimensions_are_reported(report):
    keys = [item["key"] for item in report["dimensions"]]
    assert keys == [
        "product",
        "process",
        "resource",
        "volume",
        "changeover",
        "disturbance",
    ]
    assert report["summary"]["total"] == 6


def test_every_dimension_carries_a_metric_and_chinese_evidence(report):
    for dimension in report["dimensions"]:
        assert dimension["metrics"], f"{dimension['key']} 没有任何支撑数据"
        assert dimension["headline_zh"], f"{dimension['key']} 缺少关键指标描述"
        assert dimension["evidence_zh"], f"{dimension['key']} 缺少依据说明"
        assert dimension["state"] in {FULL, PARTIAL, NONE}
        assert dimension["state_zh"]


def test_product_flexibility_counts_the_yaml_driven_products(report):
    product = next(item for item in report["dimensions"] if item["key"] == "product")
    presets = {entry["preset"] for entry in product["metrics"]["products"]}
    # D exists only as YAML and is the proof that products need no code.
    assert {"A", "B", "C", "D"} <= presets
    assert product["state"] == FULL


def test_resource_flexibility_reports_real_candidate_sets(report):
    resource = next(item for item in report["dimensions"] if item["key"] == "resource")
    assert resource["metrics"]["multi_candidate_operation_count"] > 0
    bindings = resource["metrics"]["bindings"]
    assert any(len(item["candidates"]) > 1 for item in bindings)


def test_v1_profile_reports_fewer_resource_candidates_than_v2():
    """The profile genuinely changes the decision space."""

    v1 = flexibility_report({"line_profile": "V1_STANDARD"})
    v2 = flexibility_report({"line_profile": "V2_DUAL_INSTALL"})

    def count(report_):
        entry = next(item for item in report_["dimensions"] if item["key"] == "resource")
        return entry["metrics"]["multi_candidate_operation_count"]

    assert count(v1) < count(v2)


def test_process_flexibility_stays_partial_until_a_scheduler_chooses(report):
    """OR branches are modelled; nothing selects between them yet."""

    process = next(item for item in report["dimensions"] if item["key"] == "process")
    assert process["state"] == PARTIAL
    assert process["metrics"]["alternative_operation_count"] >= 2
    branches = [branch for route in process["metrics"]["routes"] for branch in route["branches"]]
    # An unavailable branch must say why, rather than silently vanishing.
    unavailable = [item for item in branches if not item["available"]]
    assert all(item["reasons"] for item in unavailable)


def test_changeover_dimension_discloses_its_placeholder_baseline(report):
    changeover = next(item for item in report["dimensions"] if item["key"] == "changeover")
    assert changeover["state"] == PARTIAL, "占位基线下不得声称完全实现"
    baseline = changeover["metrics"]["baseline"]
    assert baseline["is_placeholder"] is True
    assert changeover["metrics"]["improvements"]["sequencing_only_ratio"] > 0.2
    assert "占位" in changeover["evidence_zh"] or "实测" in changeover["evidence_zh"]


def test_disturbance_dimension_reads_live_fault_state():
    state = {
        "line_profile": "V2_DUAL_INSTALL",
        "ui_capabilities": {"fault_injection": True},
        "faults_v2": [
            {"fault_type": "BRAZING_MISSING", "recovered": True},
            {"fault_type": "ARM_UNAVAILABLE", "recovered": False},
        ],
        "recoveries": [{"status": "RUNNING"}],
    }
    disturbance = next(
        item for item in flexibility_report(state)["dimensions"] if item["key"] == "disturbance"
    )
    assert disturbance["metrics"]["fault_count"] == 2
    assert disturbance["metrics"]["recovered_count"] == 1
    assert disturbance["metrics"]["recovery_rate"] == pytest.approx(0.5)
    assert disturbance["state"] == FULL


def test_report_works_without_a_running_line():
    """The static dimensions must not require a live snapshot."""

    report_ = flexibility_report(None)
    assert report_["summary"]["total"] == 6
    product = next(item for item in report_["dimensions"] if item["key"] == "product")
    assert product["metrics"]["product_count"] >= 3
