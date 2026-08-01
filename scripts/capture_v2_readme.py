"""Capture reproducible README frames from a real V2 three-order run.

The script advances the same ``V2BrazingApplication`` used by Viewer and
headless execution.  It does not render an idle XML pose or manually move
trays, so every image corresponds to an actual runtime/physics state.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco
from PIL import Image

from brazing_sim.dual_line.application import V2BrazingApplication
from brazing_sim.dual_line.cli import parse_args
from brazing_sim.dual_line.furnace import FurnacePhase
from brazing_sim.dual_line.tray_flow import TrayOwner

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "images" / "readme"


def _camera(
    *,
    lookat: tuple[float, float, float],
    distance: float,
    azimuth: float,
    elevation: float,
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def _save(
    renderer: mujoco.Renderer,
    application: V2BrazingApplication,
    filename: str,
    camera: mujoco.MjvCamera,
) -> None:
    renderer.update_scene(application.scene.data, camera=camera)
    frame = renderer.render()
    path = OUTPUT_DIR / filename
    Image.fromarray(frame).save(path, optimize=True)
    print(f"[capture] {filename} at t={application.runtime.sim_time:.2f}s", flush=True)


def _capture_fault_frame(
    *,
    fault_type: str,
    target: str,
    filename: str,
) -> None:
    """Capture one real defect/rework state from the shared V2 application."""

    args = parse_args(
        [
            "--headless",
            "--no-ui",
            "--orders",
            "A",
            "--fast",
            "--max-sim-time",
            "260",
        ]
    )
    application = V2BrazingApplication(args)
    application.submit_cli_orders()
    application.runtime.inject_fault(fault_type, target=target)
    model = application.scene.model
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(model, height=720, width=1280)
    try:
        while (
            not application.runtime.complete
            and application.runtime.sim_time < float(args.max_sim_time)
        ):
            application.advance_frame()
            runtime = application.runtime
            defects = tuple(runtime.faults.physical_faults.values())
            if fault_type == "BRAZING_MISSING":
                detected = next(
                    (
                        defect
                        for defect in defects
                        if defect.visual_type == "BRAZING_MISSING"
                        and defect.status == "DETECTED"
                    ),
                    None,
                )
                if detected is not None:
                    _save(
                        renderer,
                        application,
                        filename,
                        _camera(
                            lookat=(0.50, 0.00, 0.27),
                            distance=1.18,
                            azimuth=142,
                            elevation=-42,
                        ),
                    )
                    return
            else:
                recovery = next(
                    (
                        operation
                        for operation in runtime.operations.values()
                        if operation.recovery and operation.kind == "INSTALL_FIN"
                    ),
                    None,
                )
                if recovery is not None:
                    progress = application.scene.robot_motion_snapshot()[
                        recovery.resource.lower()
                    ].get("progress", 0.0)
                    if 0.28 <= float(progress) <= 0.78:
                        unit = runtime.units[recovery.unit_id]
                        assert unit.tray_id is not None
                        tray_position = application.scene.tray_position(unit.tray_id)
                        _save(
                            renderer,
                            application,
                            filename,
                            _camera(
                                lookat=tuple(float(value) for value in tray_position),
                                distance=1.32,
                                azimuth=138,
                                elevation=-38,
                            ),
                        )
                        return
        raise RuntimeError(f"V2 fault run completed without capture state: {filename}")
    finally:
        renderer.close()
        application.close()


def _capture_fault_flexibility_frames() -> None:
    _capture_fault_frame(
        fault_type="BRAZING_MISSING",
        target="path_02",
        filename="v2_fault_brazing_missing.png",
    )
    _capture_fault_frame(
        fault_type="FIN_POSE",
        target="fin_03",
        filename="v2_fault_fin_pose_rework.png",
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args(
        [
            "--headless",
            "--no-ui",
            "--orders",
            "A,A,A",
            "--max-sim-time",
            "500",
        ]
    )
    application = V2BrazingApplication(args)
    application.submit_cli_orders()
    model = application.scene.model
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(model, height=720, width=1280)
    captured: set[str] = set()
    try:
        while not application.runtime.complete and application.runtime.sim_time < float(args.max_sim_time):
            application.advance_frame()
            runtime = application.runtime
            operations = runtime.operations
            robot_motion = application.scene.robot_motion_snapshot()

            arm2 = operations.get("ARM2")
            arm2_motion = robot_motion["arm2"]
            arm2_waypoint_count = max(
                1,
                int(arm2_motion.get("waypoint_count", 0)),
            )
            arm2_path_progress = int(arm2_motion.get("waypoint_index", 0)) / arm2_waypoint_count
            if (
                "v2_dispensing_current.png" not in captured
                and arm2 is not None
                and arm2.kind == "DISPENSING"
                and 0.40 <= arm2_path_progress <= 0.70
            ):
                _save(
                    renderer,
                    application,
                    "v2_dispensing_current.png",
                    _camera(
                        lookat=(-0.30, 0.02, 0.30),
                        distance=1.80,
                        azimuth=140,
                        elevation=-30,
                    ),
                )
                captured.add("v2_dispensing_current.png")

            arm1 = operations.get("ARM1")
            arm3 = operations.get("ARM3")
            parallel_install = (
                arm1 is not None
                and arm1.kind == "INSTALL_FIN"
                and arm3 is not None
                and arm3.kind == "INSTALL_FIN"
            )
            if parallel_install and "v2_parallel_install_current.png" not in captured:
                _save(
                    renderer,
                    application,
                    "v2_parallel_install_current.png",
                    _camera(
                        lookat=(0.32, 0.10, 0.32),
                        distance=2.55,
                        azimuth=145,
                        elevation=-28,
                    ),
                )
                captured.add("v2_parallel_install_current.png")
                _save(
                    renderer,
                    application,
                    "v2_current_overview.png",
                    _camera(
                        lookat=(1.85, 0.00, 0.30),
                        distance=5.50,
                        azimuth=140,
                        elevation=-25,
                    ),
                )
                captured.add("v2_current_overview.png")

            if (
                "v2_furnace_batch_current.png" not in captured
                and runtime.furnace.state.phase is FurnacePhase.SOAK
                and sum(layer.tray_id is not None for layer in runtime.furnace.state.layers) == 3
            ):
                _save(
                    renderer,
                    application,
                    "v2_furnace_batch_current.png",
                    _camera(
                        lookat=(3.25, 0.00, 0.39),
                        distance=2.35,
                        azimuth=-40,
                        elevation=-22,
                    ),
                )
                captured.add("v2_furnace_batch_current.png")

            output_units = [
                unit
                for unit in runtime.units.values()
                if unit.tray_id is not None and runtime.flow.get(unit.tray_id).owner is TrayOwner.OUTPUT
            ]
            output_motion = next(
                (
                    motion
                    for motion in application.scene.transport_snapshot().values()
                    if motion["target"] == TrayOwner.OUTPUT.value
                ),
                None,
            )
            if (
                "v2_post_braze_output_current.png" not in captured
                and runtime.output_gate_open
                and output_units
                and output_motion is not None
                and 0.35 <= float(output_motion["progress"]) <= 0.48
            ):
                _save(
                    renderer,
                    application,
                    "v2_post_braze_output_current.png",
                    _camera(
                        lookat=(4.52, 0.00, 0.34),
                        distance=2.20,
                        azimuth=300,
                        elevation=-22,
                    ),
                )
                captured.add("v2_post_braze_output_current.png")

        expected = {
            "v2_current_overview.png",
            "v2_dispensing_current.png",
            "v2_parallel_install_current.png",
            "v2_furnace_batch_current.png",
            "v2_post_braze_output_current.png",
        }
        missing = sorted(expected - captured)
        if missing:
            raise RuntimeError(f"V2 run completed without capture states: {missing}")
    finally:
        renderer.close()
        application.close()
    _capture_fault_flexibility_frames()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
