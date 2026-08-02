"""Render compact, reproducible README animations from the real V2 runtime.

The clips are deliberately driven by operation and recovery state instead of
hard-coded frame numbers.  They therefore remain useful after timing changes
and never present a manually posed MJCF scene as a manufacturing run.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import subprocess
import tempfile

os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco
import numpy as np
from PIL import Image

from brazing_sim.dual_line.application import V2BrazingApplication
from brazing_sim.dual_line.cli import parse_args

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "images" / "readme"
WIDTH = 1280
HEIGHT = 720
ANIMATION_FPS = 15


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


def _encode_animations(frames: list[np.ndarray], destination_stem: Path) -> tuple[Path, Path]:
    """Write a high-fidelity WebP plus a broadly compatible GIF fallback."""

    if len(frames) < 2:
        raise RuntimeError(f"animation {destination_stem.name} captured too few frames")
    destination_stem.parent.mkdir(parents=True, exist_ok=True)
    webp_destination = destination_stem.with_suffix(".webp")
    gif_destination = destination_stem.with_suffix(".gif")

    # Animated WebP preserves gradients and small Chinese labels far better
    # than GIF's global 256-colour ceiling.  GitHub renders it inline, while
    # the GIF remains available as a compatibility/download fallback.
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        webp_destination,
        save_all=True,
        append_images=pil_frames[1:],
        duration=round(1000 / ANIMATION_FPS),
        loop=0,
        format="WEBP",
        quality=92,
        method=6,
        minimize_size=True,
    )

    with tempfile.TemporaryDirectory(prefix="brazing-readme-gif-") as directory:
        video = Path(directory) / "clip.mkv"
        raw = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{WIDTH}x{HEIGHT}",
                "-r",
                str(ANIMATION_FPS),
                "-i",
                "-",
                "-an",
                "-c:v",
                "ffv1",
                str(video),
            ],
            stdin=subprocess.PIPE,
        )
        assert raw.stdin is not None
        for frame in frames:
            raw.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        raw.stdin.close()
        if raw.wait() != 0:
            raise RuntimeError(
                f"failed to encode intermediate video for {destination_stem.name}"
            )

        palette = Path(directory) / "palette.png"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                "palettegen=max_colors=256:reserve_transparent=0:stats_mode=full",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-i",
                str(palette),
                "-lavfi",
                "paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                "-loop",
                "0",
                str(gif_destination),
            ],
            check=True,
        )
    return webp_destination, gif_destination


def _render_clip(
    *,
    filename: str,
    orders: str,
    camera: mujoco.MjvCamera,
    started: Callable[[V2BrazingApplication], bool],
    finished: Callable[[V2BrazingApplication], bool],
    sample_sim_seconds: float,
    maximum_clip_sim_seconds: float,
    inject_fault: tuple[str, str] | None = None,
) -> None:
    args = parse_args(
        [
            "--headless",
            "--no-ui",
            "--orders",
            orders,
            "--fast",
            "--max-sim-time",
            "320",
        ]
    )
    application = V2BrazingApplication(args)
    application.submit_cli_orders()
    if inject_fault is not None:
        application.runtime.inject_fault(inject_fault[0], target=inject_fault[1])
    application.scene.model.vis.global_.offwidth = WIDTH
    application.scene.model.vis.global_.offheight = HEIGHT
    application.scene.model.vis.quality.offsamples = 4
    renderer = mujoco.Renderer(application.scene.model, height=HEIGHT, width=WIDTH)
    frames: list[np.ndarray] = []
    clip_started_at: float | None = None
    last_sample_at = float("-inf")
    try:
        while (
            not application.runtime.complete
            and application.runtime.sim_time < float(args.max_sim_time)
        ):
            application.advance_frame()
            now = float(application.runtime.sim_time)
            if clip_started_at is None:
                if not started(application):
                    continue
                clip_started_at = now
            if now - last_sample_at + 1.0e-12 >= sample_sim_seconds:
                renderer.update_scene(application.scene.data, camera=camera)
                frames.append(renderer.render().copy())
                last_sample_at = now
            if finished(application) and len(frames) >= 2:
                break
            if now - clip_started_at >= maximum_clip_sim_seconds:
                break
        # A short hold makes the first and final physical states readable when
        # GitHub starts or loops the animation.
        if frames:
            frames = [frames[0]] * 4 + frames + [frames[-1]] * 8
        destination_stem = OUTPUT_DIR / Path(filename).stem
        webp_destination, gif_destination = _encode_animations(frames, destination_stem)
        webp_size_mib = webp_destination.stat().st_size / (1024 * 1024)
        gif_size_mib = gif_destination.stat().st_size / (1024 * 1024)
        print(
            f"[animation] {destination_stem.name}: {len(frames)} frames, "
            f"WebP={webp_size_mib:.2f} MiB, GIF={gif_size_mib:.2f} MiB, "
            f"t={clip_started_at:.2f}s",
            flush=True,
        )
    finally:
        renderer.close()
        application.close()


def _operation(application: V2BrazingApplication, resource: str):
    return application.runtime.operations.get(resource)


def _dispensing_clip() -> None:
    seen = {"started": False}

    def started(application: V2BrazingApplication) -> bool:
        operation = _operation(application, "ARM2")
        active = operation is not None and operation.kind == "DISPENSING" and not operation.recovery
        seen["started"] = seen["started"] or active
        return active

    def finished(application: V2BrazingApplication) -> bool:
        operation = _operation(application, "ARM2")
        return seen["started"] and (operation is None or operation.kind != "DISPENSING")

    _render_clip(
        filename="v2_dispensing_process.gif",
        orders="A",
        camera=_camera(
            lookat=(-0.30, 0.02, 0.30),
            distance=1.50,
            azimuth=140,
            elevation=-31,
        ),
        started=started,
        finished=finished,
        sample_sim_seconds=0.20,
        maximum_clip_sim_seconds=24.0,
    )


def _parallel_install_clip() -> None:
    def parallel(application: V2BrazingApplication) -> bool:
        arm1 = _operation(application, "ARM1")
        arm3 = _operation(application, "ARM3")
        return (
            arm1 is not None
            and arm1.kind == "INSTALL_FIN"
            and arm3 is not None
            and arm3.kind == "INSTALL_FIN"
        )

    _render_clip(
        filename="v2_parallel_install_process.gif",
        orders="A,A,A",
        camera=_camera(
            lookat=(0.32, 0.10, 0.32),
            distance=2.20,
            azimuth=145,
            elevation=-28,
        ),
        started=parallel,
        finished=lambda application: not parallel(application),
        sample_sim_seconds=0.18,
        maximum_clip_sim_seconds=16.0,
    )


def _fault_recovery_clip() -> None:
    seen = {"recovery": False}

    def started(application: V2BrazingApplication) -> bool:
        return any(
            defect.visual_type == "BRAZING_MISSING" and defect.status == "DETECTED"
            for defect in application.runtime.faults.physical_faults.values()
        )

    def finished(application: V2BrazingApplication) -> bool:
        operation = _operation(application, "ARM2")
        if operation is not None and operation.kind == "DISPENSING" and operation.recovery:
            seen["recovery"] = True
        unresolved = any(
            defect.visual_type == "BRAZING_MISSING" and defect.status != "RESOLVED"
            for defect in application.runtime.faults.physical_faults.values()
        )
        return seen["recovery"] and not unresolved

    _render_clip(
        filename="v2_fault_recovery_process.gif",
        orders="A",
        camera=_camera(
            lookat=(0.00, 0.00, 0.28),
            distance=1.95,
            azimuth=140,
            elevation=-32,
        ),
        started=started,
        finished=finished,
        sample_sim_seconds=0.30,
        maximum_clip_sim_seconds=32.0,
        inject_fault=("BRAZING_MISSING", "path_02"),
    )


def main() -> int:
    _dispensing_clip()
    _parallel_install_clip()
    _fault_recovery_clip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
