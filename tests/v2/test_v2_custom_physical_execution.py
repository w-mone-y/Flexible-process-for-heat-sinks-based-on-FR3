from __future__ import annotations

from dataclasses import replace

import pytest

from brazing_sim.api import validate_http_command
from brazing_sim.dual_line.application import V2BrazingApplication
from brazing_sim.dual_line.cli import parse_args
from brazing_sim.dual_line.unified_runtime import UnifiedV2Runtime
from brazing_sim.events import EventType
from brazing_sim.flexible import build_custom_plan


def _custom_product() -> dict[str, object]:
    return {
        "base_size_m": [0.36, 0.22, 0.008],
        "fin_size_m": [0.30, 0.002, 0.06],
        "fin_count": 6,
        "fin_pitch_m": 0.02,
        "path_margin_m": 0.015,
        "path_width_m": 0.004,
        "nozzle_spacing_m": 0.005,
        "nozzle_tip_height_m": 0.004,
        "material_speed_m_s": 0.04,
        "target_clamping_force_n": 24.0,
        "recipe": "demo_brazing",
    }


def test_v2_custom_admission_rejects_capability_parameter_before_queueing() -> None:
    valid = build_custom_plan(
        order_id="CUSTOM_BAD_CAPABILITY",
        quantity=1,
        priority=10,
        product=_custom_product(),
    )
    invalid_product = replace(valid.product, material_speed_m_s=0.01)
    invalid = replace(valid, product=invalid_product)
    runtime = UnifiedV2Runtime(fast=True)

    with pytest.raises(ValueError, match="能力/工艺路线"):
        runtime.submit_plan(invalid)

    assert runtime.physical_runtime.orders == {}
    assert runtime.physical_runtime.units == {}
    assert runtime.manufacturing_runtime.orders == {}
    assert runtime.physical_runtime.events == []


def test_v2_http_rejects_custom_quantity_outside_tray_pool() -> None:
    with pytest.raises(ValueError, match="1到3"):
        validate_http_command(
            "/orders/insert",
            {
                "line_profile": "V2_DUAL_INSTALL",
                "mode": "custom",
                "order_id": "CUSTOM_TOO_MANY_TRAYS",
                "quantity": 4,
                "priority": 10,
                "custom_product": _custom_product(),
            },
        )


def test_v2_custom_preview_accepts_a_supported_physical_route() -> None:
    preview = validate_http_command(
        "/orders/plan",
        {
            "line_profile": "V2_DUAL_INSTALL",
            "mode": "custom",
            "order_id": "CUSTOM_RELIABLE_ROUTE",
            "quantity": 1,
            "priority": 10,
            "route_strategy": "HIGH_RELIABILITY",
            "custom_product": _custom_product(),
        },
    )

    assert preview["route_strategy"] == "HIGH_RELIABILITY"
    assert any(
        task["task_type"] == "REVIEW_BRAZING_CLOSEUP"
        for task in preview["task_preview"]
    )


def test_v2_custom_order_completes_through_unified_runtime_and_mujoco() -> None:
    pytest.importorskip("mujoco")
    application = V2BrazingApplication(
        parse_args(["--headless", "--no-ui", "--fast", "--max-sim-time", "240"])
    )
    try:
        application.process_command(
            {
                "type": "order_insert",
                "mode": "custom",
                "preset": "CUSTOM",
                "order_id": "CUSTOM_PHYSICAL_E2E",
                "quantity": 1,
                "priority": 20,
                "route_strategy": "STANDARD",
                "custom_product": _custom_product(),
            }
        )
        assert application.last_error == ""

        for _ in range(5_000):
            application.advance_frame()
            if application.runtime.complete and application.scene.transport_settled:
                break
        else:
            raise AssertionError("custom V2 order did not complete in the MuJoCo actor")

        snapshot = application.runtime.snapshot()
        assert snapshot["execution_mode"] == "UNIFIED_PHYSICAL_RUNTIME"
        assert snapshot["orders"][0]["mode"] == "custom"
        assert snapshot["units"][0]["product_id"] == "CUSTOM_CUSTOM_PHYSICAL_E2E"
        assert snapshot["units"][0]["fin_count"] == 6
        assert snapshot["units"][0]["stage"] == "COMPLETE"
        assert snapshot["physical_execution_complete"] is True
        assert snapshot["physical_completion_gates"]["failed_checks"] == []

        physical_evidence = [
            event.payload.get("metrics", {}).get("physical_completion", {})
            for event in application.runtime.manufacturing_runtime.events.history
            if event.event_type is EventType.TASK_SUCCEEDED
        ]
        assert any(
            evidence.get("source") == "mujoco:v2_execution_gate"
            for evidence in physical_evidence
        )
    finally:
        application.close()
