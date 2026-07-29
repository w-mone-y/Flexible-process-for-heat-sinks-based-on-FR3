from __future__ import annotations

import pytest

from brazing_sim.dual_line import (
    BatchRecipe,
    DualInstallDispatcher,
    DualLineTopology,
    FurnacePhase,
    InstallBranch,
    InstallRequest,
    InstallResourceState,
    ThroughBatchFurnace,
    TrayFlowController,
    TrayOwner,
    TrayPhase,
)


def test_dual_line_topology_has_two_install_branches_and_one_forward_flow() -> None:
    topology = DualLineTopology.standard()

    assert topology.station("S1_BASE_LOADING").world_xyz[:2] == (-0.55, 0.35)
    assert topology.successors("S2B_MATERIAL_INSPECTION") == (
        "S3A_ARM1_INSTALL",
        "S3B_ARM3_INSTALL",
    )
    assert topology.successors("S3A_ARM1_INSTALL") == ("MERGE_A_WAIT",)
    assert topology.successors("S3B_ARM3_INSTALL") == ("MERGE_B_WAIT",)
    assert topology.successors("MERGE_A_WAIT") == ("Y_MERGE_SHARED",)
    assert topology.successors("MERGE_B_WAIT") == ("Y_MERGE_SHARED",)
    assert topology.successors("Y_MERGE_SHARED") == ("S4_PRE_BRAZE_INSPECTION",)
    assert topology.route("S1_BASE_LOADING", "FINISHED_OUTPUT")[0] == "S1_BASE_LOADING"
    assert topology.route("S1_BASE_LOADING", "FINISHED_OUTPUT")[-1] == "FINISHED_OUTPUT"
    assert topology.validate() == ()


def test_dispatcher_selects_earliest_finish_and_respects_arm3_inspection_priority() -> None:
    dispatcher = DualInstallDispatcher()
    request = InstallRequest(
        tray_id="V2_TRAY_01",
        fin_count=7,
        ready_at=10.0,
        due_at=80.0,
        priority=10,
    )
    resources = (
        InstallResourceState(
            branch=InstallBranch.ARM1_A,
            available_at=12.0,
            seconds_per_fin=2.0,
        ),
        InstallResourceState(
            branch=InstallBranch.ARM3_B,
            available_at=10.0,
            seconds_per_fin=1.5,
            inspection_reservations=((13.0, 25.0),),
        ),
    )

    decision = dispatcher.assign(request, resources)

    assert decision.branch is InstallBranch.ARM1_A
    assert decision.candidates[InstallBranch.ARM3_B].inspection_wait_s > 0.0
    assert "检测预约" in decision.explanation_zh


def test_dispatcher_uses_arm3_when_camera_has_a_genuine_idle_window() -> None:
    decision = DualInstallDispatcher().assign(
        InstallRequest("V2_TRAY_02", fin_count=4, ready_at=0.0, priority=20),
        (
            InstallResourceState(
                branch=InstallBranch.ARM1_A,
                available_at=20.0,
                seconds_per_fin=2.0,
            ),
            InstallResourceState(
                branch=InstallBranch.ARM3_B,
                available_at=0.0,
                seconds_per_fin=1.5,
                inspection_reservations=((20.0, 25.0),),
            ),
        ),
    )

    assert decision.branch is InstallBranch.ARM3_B
    assert decision.finish_at == pytest.approx(6.0)


def test_six_trays_have_one_owner_and_virtual_return_is_explicit() -> None:
    flow = TrayFlowController(capacity=6)
    tray = flow.assign_order("ORDER_A", "ORDER_A_UNIT_01", now=1.0)

    assert tray.tray_id == "V2_TRAY_01"
    assert tray.phase is TrayPhase.BASE_LOADING
    assert tray.owner is TrayOwner.S1
    flow.handoff(tray.tray_id, TrayOwner.S1, TrayOwner.S2A, TrayPhase.DISPENSING, now=2.0)
    with pytest.raises(RuntimeError, match="ownership mismatch"):
        flow.handoff(tray.tray_id, TrayOwner.S1, TrayOwner.S2B, TrayPhase.MATERIAL_INSPECTION, now=3.0)

    flow.handoff(tray.tray_id, TrayOwner.S2A, TrayOwner.S2B, TrayPhase.MATERIAL_INSPECTION, now=3.0)
    flow.handoff(tray.tray_id, TrayOwner.S2B, TrayOwner.INSTALL_A, TrayPhase.FIN_INSTALLATION, now=4.0)
    flow.handoff(
        tray.tray_id,
        TrayOwner.INSTALL_A,
        TrayOwner.MERGE_A_WAIT,
        TrayPhase.MERGE_WAIT,
        now=5.0,
    )
    flow.handoff(
        tray.tray_id,
        TrayOwner.MERGE_A_WAIT,
        TrayOwner.MERGE,
        TrayPhase.MERGING,
        now=6.0,
    )
    flow.handoff(tray.tray_id, TrayOwner.MERGE, TrayOwner.S4, TrayPhase.PRE_BRAZE_INSPECTION, now=7.0)
    flow.handoff(tray.tray_id, TrayOwner.S4, TrayOwner.BUFFER_1, TrayPhase.FURNACE_BUFFER, now=8.0)
    flow.handoff(tray.tray_id, TrayOwner.BUFFER_1, TrayOwner.FURNACE, TrayPhase.BRAZING, now=9.0)
    flow.handoff(
        tray.tray_id,
        TrayOwner.FURNACE,
        TrayOwner.POST_SCAN,
        TrayPhase.POST_BRAZE_INSPECTION,
        now=10.0,
    )
    flow.handoff(tray.tray_id, TrayOwner.POST_SCAN, TrayOwner.OUTPUT, TrayPhase.DELIVERED, now=11.0)
    flow.mark_product_removed(tray.tray_id, now=12.0)
    flow.start_virtual_return(tray.tray_id, now=13.0)
    assert flow.get(tray.tray_id).phase is TrayPhase.VIRTUAL_RETURN
    flow.complete_virtual_return(tray.tray_id, now=14.0)
    returned = flow.get(tray.tray_id)
    assert returned.phase is TrayPhase.EMPTY_BUFFER
    assert returned.owner is TrayOwner.EMPTY_BUFFER
    assert returned.order_id is None


def test_furnace_requires_locked_layers_and_uses_front_then_rear_door() -> None:
    furnace = ThroughBatchFurnace(capacity=3, demo_cycle_seconds=30.0)
    recipe = BatchRecipe("CAB_A", "aluminium", 600.0, 240.0, 0.10)
    furnace.plan_batch(
        (("V2_TRAY_01", recipe), ("V2_TRAY_02", recipe)),
        now=0.0,
    )
    furnace.open_front(now=0.0)
    furnace.load_front("V2_TRAY_01", layer=0, now=1.0)
    furnace.lock_layer(0, now=1.1)
    furnace.load_front("V2_TRAY_02", layer=1, now=2.0)
    furnace.lock_layer(1, now=2.1)

    with pytest.raises(RuntimeError, match="front door"):
        furnace.start_cycle(now=3.0)
    furnace.close_front(now=3.0)
    furnace.start_cycle(now=3.1)
    furnace.update(33.1)
    assert furnace.ready_to_unload
    assert furnace.state.real_equivalent_elapsed_s == pytest.approx(3600.0)

    furnace.open_rear(now=33.2)
    assert furnace.unload_rear(now=33.3) == "V2_TRAY_02"
    assert furnace.unload_rear(now=33.4) == "V2_TRAY_01"
    assert furnace.state.rear_door_open
    assert furnace.state.phase is FurnacePhase.UNLOADING
    assert not furnace.state.complete
    furnace.close_rear(now=33.5)
    assert not furnace.state.rear_door_open
    assert furnace.state.complete


def test_furnace_rejects_an_incompatible_cross_order_batch() -> None:
    furnace = ThroughBatchFurnace()
    standard = BatchRecipe("CAB_A", "aluminium", 600.0, 240.0, 0.10)
    incompatible = BatchRecipe("CAB_B", "aluminium", 610.0, 240.0, 0.10)

    with pytest.raises(ValueError, match="incompatible"):
        furnace.plan_batch(
            (("V2_TRAY_01", standard), ("V2_TRAY_02", incompatible)),
            now=0.0,
        )
