"""Raw-fin magazine layout regression tests."""

from __future__ import annotations

import pytest

from brazing_sim.layout import SHALLOW_U_LAYOUT


def _positions(count: int) -> list[tuple[float, float, float]]:
    return [
        SHALLOW_U_LAYOUT.raw_fin_position(
            index,
            count,
            table_top_z=0.10,
            fin_height_m=0.06,
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("count", (7, 8, 9, 10, 11, 12))
def test_dense_orders_keep_every_raw_fin_on_one_tier(count: int) -> None:
    positions = _positions(count)
    assert all(position[2] == pytest.approx(0.28) for position in positions)
    assert positions[-1][1] - positions[0][1] <= 0.190000001


def test_b_order_retains_wide_gripper_clearance() -> None:
    positions = _positions(4)
    spacings = [positions[index + 1][1] - positions[index][1] for index in range(3)]
    assert spacings == pytest.approx([0.075, 0.075, 0.075])


def test_c_order_is_single_tier_and_more_compact_than_b() -> None:
    positions = _positions(7)
    spacings = [positions[index + 1][1] - positions[index][1] for index in range(6)]
    assert spacings == pytest.approx([0.031] * 6)
