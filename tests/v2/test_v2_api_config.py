from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from brazing_sim.api import SharedState, post_json, start_http_server, validate_http_command
from brazing_sim.fault_catalog import MANUAL_FAULT_CATALOG
from brazing_sim.flexible import load_order_plans
from brazing_sim.manufacturing_config import (
    load_fault_scenario,
    load_resource_config,
    load_scheduler_config,
)

ROOT = Path(__file__).resolve().parents[2]


def _custom_product(*, fin_count: int = 6) -> dict[str, object]:
    return {
        "base_size_m": [0.36, 0.22, 0.008],
        "fin_size_m": [0.30, 0.002, 0.06],
        "fin_count": fin_count,
        "fin_pitch_m": 0.02,
        "path_margin_m": 0.015,
        "path_width_m": 0.004,
        "nozzle_spacing_m": 0.005,
        "nozzle_tip_height_m": 0.004,
        "material_speed_m_s": 0.04,
        "target_clamping_force_n": 24.0,
        "recipe": "demo_brazing",
    }


def test_v2_configuration_and_multi_order_files_load() -> None:
    scheduler = load_scheduler_config(ROOT / "config" / "scheduler.yaml")
    resources, zones = load_resource_config(ROOT / "config" / "resources.yaml")
    scenario = load_fault_scenario(ROOT / "config" / "faults" / "scenario_01.yaml")
    plans = load_order_plans(ROOT / "config" / "orders" / "batch_abc.yaml")
    assert scheduler.max_assignments_per_tick == 3
    assert {resource.resource_id for resource in resources} >= {"ARM1", "ARM2", "ARM3", "FURNACE"}
    assert "ZONE_TABLE2_CORE" in zones
    assert scenario.random_seed == 42
    assert [plan.product.preset for plan in plans] == ["A", "C", "B"]


def test_order_plan_and_v2_control_commands_validate() -> None:
    preview = validate_http_command(
        "/orders/plan",
        {"order_id": "UI_TEST", "preset": "C", "quantity": 2, "priority": 20},
    )
    assert preview["type"] == "order_plan"
    assert preview["plan"]["fin_count"] == 7
    assert preview["plan"]["estimated_task_count"] == len(preview["task_preview"])
    assert validate_http_command("/resources/arm2/fault", {})["resource_id"] == "ARM2"
    assert validate_http_command("/resources/arm2/recover", {})["type"] == "resource_recover"
    assert validate_http_command("/scheduler/replan", {})["type"] == "scheduler_replan"
    fault = validate_http_command(
        "/faults/inject",
        {
            "fault_type": "arm_unavailable",
            "target": "arm2",
            "severity": "recoverable",
            "auto_recover": True,
            "duration_s": 8,
        },
    )
    assert fault == {
        "type": "manual_fault_inject",
        "fault_type": "ARM_UNAVAILABLE",
        "target": "ARM2",
        "severity": "recoverable",
        "auto_recover": False,
        "duration_s": None,
        "label_zh": "机械臂暂时离线",
    }
    safety = validate_http_command(
        "/faults/inject",
        {
            "fault_type": "CONTACT_SAFETY_STOP",
            "auto_recover": True,
            "duration_s": 8,
        },
    )
    assert safety["auto_recover"] is False
    assert safety["duration_s"] is None
    assert MANUAL_FAULT_CATALOG["CONTACT_SAFETY_STOP"].simulated_manual_review
    assert MANUAL_FAULT_CATALOG["TRAY_STATE_INCONSISTENT"].simulated_manual_review
    assert {"FIN_POSE", "BRAZING_MISSING", "FURNACE_PROFILE", "FORK_TIMEOUT"} <= set(MANUAL_FAULT_CATALOG)


def test_custom_v2_order_can_be_previewed_and_inserted() -> None:
    payload = {
        "line_profile": "V2_DUAL_INSTALL",
        "mode": "custom",
        "order_id": "CUSTOM_UI",
        "quantity": 2,
        "priority": 18,
        "custom_product": _custom_product(fin_count=6),
    }

    preview = validate_http_command("/orders/plan", payload)
    insertion = validate_http_command("/orders/insert", payload)

    assert preview["plan"]["preset"] == "CUSTOM"
    assert preview["plan"]["fin_count"] == 6
    assert preview["plan"]["path_count"] == 12
    assert preview["plan"]["comb_module"] == "comb_insert_20mm"
    assert insertion["type"] == "order_insert"
    assert insertion["custom_product"] == payload["custom_product"]


def test_v2_custom_preview_rejects_geometry_outside_the_physical_tray() -> None:
    product = _custom_product()
    product["base_size_m"] = [0.44, 0.22, 0.008]
    with pytest.raises(ValueError, match="390×250 mm"):
        validate_http_command(
            "/orders/plan",
            {
                "line_profile": "V2_DUAL_INSTALL",
                "mode": "custom",
                "order_id": "TOO_LARGE",
                "custom_product": product,
            },
        )


def test_post_json_exposes_custom_order_validation_reason_from_http_400() -> None:
    shared = SharedState()
    server = start_http_server(shared, "127.0.0.1", 0)
    product = _custom_product()
    product["base_size_m"] = [0.44, 0.22, 0.008]
    payload = {
        "line_profile": "V2_DUAL_INSTALL",
        "mode": "custom",
        "order_id": "TOO_LARGE_OVER_HTTP",
        "custom_product": product,
    }
    url = f"http://127.0.0.1:{server.server_address[1]}/orders/insert"
    try:
        with pytest.raises(ValueError, match="390×250 mm"):
            post_json(url, payload)
    finally:
        server.shutdown()
        server.server_close()


def test_removed_fin_insert_fault_is_not_exposed_or_accepted() -> None:
    assert "FIN_INSERT_FAILED" not in MANUAL_FAULT_CATALOG
    with pytest.raises(ValueError, match="unknown manual fault type"):
        validate_http_command(
            "/faults/inject",
            {"fault_type": "FIN_INSERT_FAILED", "target": "fin_02"},
        )


def test_v2_get_views_use_shared_snapshot() -> None:
    shared = SharedState()
    shared.update(
        scheduler={"mode": "DYNAMIC_PRIORITY"},
        tasks=[{"task_id": "t1"}],
        resources_v2={"ARM1": {"status": "IDLE"}},
        orders=[{"order_id": "O1"}],
    )
    server = start_http_server(shared, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert (
            json.loads(urlopen(base + "/scheduler/status").read())["mode"] == "DYNAMIC_PRIORITY"
        )  # noqa: S310
        assert json.loads(urlopen(base + "/tasks").read())["tasks"][0]["task_id"] == "t1"  # noqa: S310
        catalog = json.loads(urlopen(base + "/fault-catalog").read())["faults"]  # noqa: S310
        assert len(catalog) == len(MANUAL_FAULT_CATALOG)
        assert all(item["label_zh"] for item in catalog)
        request = Request(
            base + "/orders/insert",
            data=json.dumps({"order_id": "O2", "preset": "A", "quantity": 1}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        assert json.loads(urlopen(request).read())["type"] == "order_insert"  # noqa: S310
        assert shared.commands.get(timeout=1)["order_id"] == "O2"
    finally:
        server.shutdown()
        server.server_close()
