"""Capture the physical ordinary-order chain after the final fin."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from brazing_line import BrazingApplication, parse_args
from brazing_sim.planning.task_models import TaskType


ROOT = Path(__file__).resolve().parent / "post_fin"


def capture(app, renderer, name, label, lookat, distance, azimuth, elevation=-35.0):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(lookat, dtype=float)
    camera.distance = float(distance)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    mujoco.mj_forward(app.scene.model, app.scene.data)
    renderer.update_scene(app.scene.data, camera=camera)
    image = Image.fromarray(np.asarray(renderer.render()).copy())
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((16, 14, 350, 48), radius=8, fill=(8, 14, 22, 225))
    draw.text((27, 23), label, fill=(255, 255, 255))
    image.save(ROOT / f"{name}.png")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in ROOT.glob("*.png"):
        path.unlink()
    app = BrazingApplication(
        parse_args(
            [
                "--headless",
                "--dt",
                "0.02",
                "--no-ui",
                "--no-terminal-commands",
                "--port",
                "0",
            ]
        )
    )
    renderer = mujoco.Renderer(app.scene.model, height=480, width=640)
    captured: set[str] = set()
    try:
        app.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "A",
                "order_id": "POST_FIN_VISUAL",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        for _ in range(16_000):
            app.tick()
            registry = app.scene.registry
            tasks = list(app.manufacturing_runtime.graph)

            def succeeded(kind):
                return any(t.task_type is kind and t.status.value == "SUCCEEDED" for t in tasks)

            if succeeded(TaskType.LOCK_FIXTURE) and "01_press_locked" not in captured:
                capture(app, renderer, "01_press_locked", "01  FINS INSPECTED + PRESS LOCKED", (0.48, 0.0, 0.27), 0.78, 125)
                captured.add("01_press_locked")
            outfeed = registry.batch_joint_position("batch_outfeed_joint")
            if 0.35 < outfeed < 0.65 and "02_conveyor_in" not in captured:
                capture(app, renderer, "02_conveyor_in", "02  PALLET MOVING INTO FURNACE", (0.75, 0.38, 0.34), 1.20, -90)
                captured.add("02_conveyor_in")
            if succeeded(TaskType.LOCK_RACK_LAYER) and "03_rack_locked" not in captured:
                capture(app, renderer, "03_rack_locked", "03  PALLET LOCKED IN FURNACE RACK", (0.75, 0.84, 0.40), 1.05, -90)
                captured.add("03_rack_locked")
            furnace = next((t for t in tasks if t.task_type is TaskType.RUN_FURNACE), None)
            if (
                furnace is not None
                and furnace.status.value == "RUNNING"
                and registry.batch_joint_position("furnace_door_joint") < 0.03
                and "04_furnace_cycle" not in captured
            ):
                capture(app, renderer, "04_furnace_cycle", "04  10 s BRAZING CYCLE - DOOR CLOSED", (0.75, 0.84, 0.43), 1.15, -95)
                captured.add("04_furnace_cycle")
            if succeeded(TaskType.UNLOAD_RACK_LAYER) and "05_unloaded" not in captured:
                capture(app, renderer, "05_unloaded", "05  PALLET AT POST-BRAZE INSPECTION", (0.75, -0.10, 0.26), 0.85, 90)
                captured.add("05_unloaded")
            post = next((t for t in tasks if t.task_type is TaskType.POST_BRAZE_INSPECTION), None)
            if post is not None and post.status.value == "RUNNING" and "06_post_inspection" not in captured:
                if app.scene.arms["arm3"].current_tcp_pose().position[0] > 0.52:
                    capture(app, renderer, "06_post_inspection", "06  ARM3 POST-BRAZE SCAN", (0.68, -0.08, 0.32), 0.95, 100)
                    captured.add("06_post_inspection")
            output = registry.batch_joint_position("batch_output_joint")
            gate = registry.finished_output_gate_fraction
            if output > 0.55 and gate > 0.95 and "07_delivery_enter" not in captured:
                capture(app, renderer, "07_delivery_enter", "07  LOADED PALLET ENTERING OUTPUT", (0.75, -0.80, 0.30), 1.25, 90)
                captured.add("07_delivery_enter")
            base_alpha = app.scene.model.geom_rgba[registry.geom_id("batch_tray_01_base"), 3]
            tray_alpha = app.scene.model.geom_rgba[registry.geom_id("batch_tray_01_geom"), 3]
            if base_alpha < 0.1 and tray_alpha > 0.9 and output > 0.25 and "08_empty_return" not in captured:
                capture(app, renderer, "08_empty_return", "08  PRODUCT HANDED OFF - EMPTY PALLET RETURNS", (0.75, -0.80, 0.30), 1.25, 90)
                captured.add("08_empty_return")
            if app.manufacturing_runtime.terminal:
                capture(app, renderer, "09_complete", "09  DELIVERY COMPLETE - GATE CLOSED", (0.75, -0.80, 0.30), 1.25, 90)
                captured.add("09_complete")
                break
        required = {f"{index:02d}" for index in range(1, 10)}
        actual = {path.stem.split("_", 1)[0] for path in ROOT.glob("*.png")}
        if actual != required:
            raise RuntimeError(f"missing post-fin frames: {sorted(required - actual)}")
        images = [Image.open(path).convert("RGB") for path in sorted(ROOT.glob("*.png"))]
        sheet = Image.new("RGB", (1280, 1440), (20, 24, 30))
        for index, image in enumerate(images):
            sheet.paste(image.resize((384, 288)), ((index % 2) * 640 + 128, (index // 2) * 288))
        sheet.save(ROOT.parent / "post_fin_contact_sheet.png")
        print("captured", len(images), "post-fin physical milestones")
    finally:
        renderer.close()
        app.close()


if __name__ == "__main__":
    main()
