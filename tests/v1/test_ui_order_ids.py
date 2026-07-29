from __future__ import annotations

from brazing_sim.ui import unique_order_id


def test_ui_order_id_advances_past_completed_and_pending_orders() -> None:
    assert unique_order_id("UI_ORDER_001", {"UI_ORDER_001"}) == "UI_ORDER_002"
    assert (
        unique_order_id(
            "UI_ORDER_001",
            {"UI_ORDER_001", "UI_ORDER_002", "UI_ORDER_003"},
        )
        == "UI_ORDER_004"
    )


def test_ui_order_id_preserves_custom_names_and_fills_empty_input() -> None:
    assert unique_order_id("CUSTOM", {"CUSTOM"}) == "CUSTOM_001"
    assert unique_order_id("ORDER_009", {"ORDER_009"}) == "ORDER_010"
    assert unique_order_id("", set()) == "UI_ORDER_001"
