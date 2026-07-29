from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from brazing_line_cinematic import CINEMATIC_XML, cinematic_args

ROOT = Path(__file__).resolve().parents[2]


def test_cinematic_entry_preserves_explicit_xml_override() -> None:
    assert cinematic_args(["--order", "A"])[:2] == ["--xml", str(CINEMATIC_XML)]
    custom = ["--xml", "custom.xml", "--order", "B"]
    assert cinematic_args(custom) == custom


def test_cinematic_scene_inherits_complete_standard_contract() -> None:
    standard = mujoco.MjModel.from_xml_path(str(ROOT / "scenes" / "production" / "brazing_line.xml"))
    cinematic = mujoco.MjModel.from_xml_path(str(CINEMATIC_XML))

    assert cinematic.nbody > standard.nbody
    assert cinematic.ngeom > standard.ngeom
    assert cinematic.nlight > standard.nlight
    assert cinematic.ncam > standard.ncam
    for name in (
        "arm1_base",
        "arm2_base",
        "arm3_base",
        "assembly_tray",
        "furnace",
        "finished_output_gate",
    ):
        assert int(cinematic.body(name).id) >= 0
    for name in (
        "conveyor_slide_joint",
        "batch_outfeed_joint",
        "batch_output_joint",
        "finished_output_gate_joint",
    ):
        assert int(cinematic.joint(name).id) >= 0


def test_cinematic_overlay_is_visual_only_and_high_fidelity() -> None:
    standard = mujoco.MjModel.from_xml_path(str(ROOT / "scenes" / "production" / "brazing_line.xml"))
    model = mujoco.MjModel.from_xml_path(str(CINEMATIC_XML))
    quality = model.vis.quality
    assert int(quality.shadowsize) == 4096
    assert int(quality.offsamples) == 4
    assert int(quality.numslices) == 32
    assert int(quality.numstacks) == 16
    assert int(model.vis.global_.offwidth) >= 1600
    assert int(model.vis.global_.offheight) >= 1000

    cinematic_geoms = [
        model.geom(index)
        for index in range(model.ngeom)
        if (model.geom(index).name or "").startswith("cinematic_")
    ]
    # Refine the machines themselves instead of surrounding them with unrelated
    # factory props.  The overlay remains visual-only so physics is unchanged.
    assert model.ngeom - standard.ngeom >= 200
    assert len(cinematic_geoms) >= 200
    assert all(int(geom.contype[0]) == 0 for geom in cinematic_geoms)
    assert all(int(geom.conaffinity[0]) == 0 for geom in cinematic_geoms)
    for name in (
        "cinematic_furnace_thermocouple",
        "cinematic_furnace_brick_l1",
        "cinematic_out_roller_07",
        "cinematic_box_photoeye_tx",
        "cinematic_door_inner_gasket_top",
        "cinematic_gate_guide_shoe_front",
        "cinematic_dispenser_needle_valve_left",
    ):
        assert int(model.geom(name).id) >= 0

    for name in (
        "cinematic_factory_shell",
        "cinematic_utility_bank",
        "cinematic_safety_fence",
        "cinematic_floor_graphics",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) == -1

    for name in (
        "cinematic_furnace_door_detail_weld",
        "cinematic_output_gate_detail_weld",
        "cinematic_dispenser_detail_weld",
    ):
        assert int(model.equality(name).id) >= 0
    for camera in (
        "cinematic_overview",
        "cinematic_process_closeup",
        "cinematic_furnace_closeup",
        "cinematic_output_closeup",
    ):
        assert int(model.camera(camera).id) >= 0


def test_moving_equipment_details_follow_their_physical_bodies() -> None:
    model = mujoco.MjModel.from_xml_path(str(CINEMATIC_XML))
    data = mujoco.MjData(model)
    pairs = (
        ("furnace_door", "cinematic_furnace_door_detail", "furnace_door_joint"),
        (
            "finished_output_gate",
            "cinematic_output_gate_detail",
            "finished_output_gate_joint",
        ),
    )

    for _, _, joint_name in pairs:
        joint = model.joint(joint_name)
        low, high = (float(value) for value in joint.range)
        data.qpos[int(joint.qposadr[0])] = low + 0.72 * (high - low)
    for _ in range(600):
        mujoco.mj_step(model, data)

    for physical_name, detail_name, _ in pairs:
        physical = data.body(physical_name)
        detail = data.body(detail_name)
        assert float(np.linalg.norm(detail.xpos - physical.xpos)) < 1e-6
        assert float(abs(np.dot(detail.xquat, physical.xquat))) > 1.0 - 1e-9
