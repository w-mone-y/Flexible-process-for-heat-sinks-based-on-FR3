"""Capture every cumulative A-order bead and fin installation milestone."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from brazing_line import BrazingApplication, parse_args
from brazing_sim.planning.task_models import TaskType


ROOT = Path(__file__).resolve().parent
MATERIAL_DIR = ROOT / "material"
FIN_DIR = ROOT / "fins"


def _camera_for_tray(application: BrazingApplication) -> mujoco.MjvCamera:
    body_id = application.scene.registry.body_id("batch_tray_01")
    centre = np.asarray(application.scene.data.xpos[body_id], dtype=float)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = centre + np.asarray([0.0, 0.0, 0.035])
    camera.distance = 0.64
    # View from the side opposite the active robot.  This keeps the nozzle or
    # gripper out of the product centre after its physical retreat stage.
    camera.azimuth = 90.0
    camera.elevation = -62.0
    return camera


def _capture(
    application: BrazingApplication,
    renderer: mujoco.Renderer,
    target: Path,
    label: str,
) -> None:
    mujoco.mj_forward(application.scene.model, application.scene.data)
    renderer.update_scene(application.scene.data, camera=_camera_for_tray(application))
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_REFLECTION)] = 0
    image = Image.fromarray(np.asarray(renderer.render()).copy())
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 16, 210, 50), radius=8, fill=(10, 16, 25, 220))
    draw.text((30, 25), label, fill=(255, 255, 255))
    image.save(target)


def _sheet(files: list[Path], target: Path, columns: int) -> None:
    images = [Image.open(path).convert("RGB") for path in files]
    thumb_size = (480, 300)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), (22, 26, 32))
    for index, image in enumerate(images):
        image.thumbnail(thumb_size)
        x = (index % columns) * thumb_size[0] + (thumb_size[0] - image.width) // 2
        y = (index // columns) * thumb_size[1] + (thumb_size[1] - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(target)


def main() -> None:
    MATERIAL_DIR.mkdir(parents=True, exist_ok=True)
    FIN_DIR.mkdir(parents=True, exist_ok=True)
    for path in (*MATERIAL_DIR.glob("*.png"), *FIN_DIR.glob("*.png")):
        path.unlink()

    args = parse_args(
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
    application = BrazingApplication(args)
    renderer = mujoco.Renderer(application.scene.model, height=480, width=640)
    captured_paths: set[int] = set()
    captured_fins: set[int] = set()
    pending_paths: set[int] = set()
    try:
        application.process_command(
            {
                "type": "order_insert",
                "mode": "preset",
                "preset": "A",
                "order_id": "VISUAL_AUDIT_A",
                "quantity": 1,
                "priority": 10,
                "route_strategy": "STANDARD",
                "urgent": False,
            }
        )
        for _ in range(14_000):
            application.tick()
            registry = application.scene.registry
            progress = [registry._batch_path_progress.get((0, index), 0.0) for index in range(10)]
            for index, amount in enumerate(progress):
                if amount >= 0.999 and index not in captured_paths:
                    pending_paths.add(index)

            # Capture only after the just-completed pass has lifted clear.
            # Each physical pass has two paths because Arm2 uses two nozzles.
            arm2_skill = next(
                (
                    execution.skill
                    for execution in application.manufacturing_runtime.executor.active.values()
                    if execution.resource_id == "ARM2"
                    and execution.task.task_type is TaskType.DISPENSE_BRAZING
                ),
                None,
            )
            arm2_label = ""
            if arm2_skill is not None and arm2_skill.arm_stages:
                arm2_label = arm2_skill.arm_stages[arm2_skill.arm_stage_index].label
            safe_to_capture_paths = bool(pending_paths) and (
                "双喷嘴接近" in arm2_label or "喷枪安全撤离" in arm2_label
            )
            if safe_to_capture_paths:
                for index in sorted(pending_paths):
                    captured_paths.add(index)
                    _capture(
                        application,
                        renderer,
                        MATERIAL_DIR / f"path_{index + 1:02d}.png",
                        f"BRAZING PATH {index + 1:02d}/10",
                    )
                pending_paths.clear()

            for index in range(5):
                fin_id = f"fin_{index + 1:02d}"
                installed = any(
                    task.task_type is TaskType.INSTALL_FIN
                    and task.tray_id == "tray_01"
                    and task.payload.get("fin_id") == fin_id
                    and task.status.value == "SUCCEEDED"
                    for task in application.manufacturing_runtime.graph
                )
                if installed and index not in captured_fins:
                    captured_fins.add(index)
                    _capture(
                        application,
                        renderer,
                        FIN_DIR / f"fin_{index + 1:02d}.png",
                        f"FIN INSTALLED {index + 1:02d}/05",
                    )
            if len(captured_paths) == 10 and len(captured_fins) == 5:
                break
        else:
            raise RuntimeError(
                f"visual audit incomplete: paths={sorted(captured_paths)}, fins={sorted(captured_fins)}"
            )

        material_files = sorted(MATERIAL_DIR.glob("path_*.png"))
        fin_files = sorted(FIN_DIR.glob("fin_*.png"))
        _sheet(material_files, ROOT / "material_contact_sheet.png", columns=2)
        _sheet(fin_files, ROOT / "fin_contact_sheet.png", columns=1)
        print(f"captured {len(material_files)} material frames and {len(fin_files)} fin frames")
    finally:
        renderer.close()
        application.close()


if __name__ == "__main__":
    main()
