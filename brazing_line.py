"""Three-FR3 heat-sink brazing-line simulation entry point."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from brazing_sim.actors import build_scene_actors
from brazing_sim.api import SharedState, rgb_frame_to_ppm, start_http_server, start_terminal_thread
from brazing_sim.domain import OrderStage
from brazing_sim.motion import MotionConfig, Pose
from brazing_sim.process import ProcessCoordinator
from brazing_sim.safety import ContactMonitor
from brazing_sim.scene import BrazingScene
from brazing_sim.ui import run_ui_client

ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-FR3 heat-sink brazing simulation")
    parser.add_argument("--ui-client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--order", choices=("A",), default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--fast", action="store_true", help="skip robot travel while preserving process state"
    )
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--no-terminal-commands", action="store_true")
    parser.add_argument("--show-mujoco-ui", action="store_true")
    parser.add_argument("--xml", default=str(ROOT / "brazing_line.xml"))
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--max-sim-time", type=float, default=1800.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=8.0)
    parser.add_argument(
        "--fault",
        action="append",
        default=[],
        metavar="TYPE[:TARGET[:SEVERITY]]",
        help="arm a deterministic fault before starting the order",
    )
    args = parser.parse_args(argv)
    if args.dt <= 0 or args.max_sim_time <= 0:
        parser.error("--dt and --max-sim-time must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0:
        parser.error("camera dimensions and FPS must be positive")
    return args


def parse_fault_option(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    fault_type = parts[0].strip().lower()
    target = parts[1].strip() if len(parts) > 1 else ""
    severity = parts[2].strip().lower() if len(parts) > 2 and parts[2].strip() else "recoverable"
    if fault_type == "furnace_profile" and target in {"recoverable", "severe"} and len(parts) == 2:
        severity, target = target, ""
    return fault_type, target, severity


class BrazingApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        motion = MotionConfig(dt=float(args.dt))
        self.scene = BrazingScene(args.xml, order="A", motion_config=motion, raw=True)
        self.scene.model.opt.timestep = float(args.dt)
        self.coordinator: ProcessCoordinator | None = None
        actors = build_scene_actors(
            self.scene,
            lambda: None if self.coordinator is None else self.coordinator.product,
            fast=bool(args.fast),
        )
        self.coordinator = ProcessCoordinator(actors=actors, fast=bool(args.fast))
        self.shared = SharedState()
        self.server = start_http_server(self.shared, args.host, int(args.port))
        self.actual_port = int(self.server.server_address[1])
        self.safety = ContactMonitor(self.scene.model)
        self.ui_process: subprocess.Popen[Any] | None = None
        self.visualized_faults: set[int] = set()
        self.last_stage = ""
        self.last_publish_wall = 0.0
        self.last_camera_wall = 0.0
        self.running = True

    def start_services(self) -> None:
        print(f"[HTTP] http://{self.args.host}:{self.actual_port}", flush=True)
        if not self.args.no_terminal_commands:
            start_terminal_thread(self.shared)
        if not self.args.headless and not self.args.no_ui:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--ui-client",
                "--host",
                str(self.args.host),
                "--port",
                str(self.actual_port),
            ]
            self.ui_process = subprocess.Popen(command)

    def arm_faults(self) -> None:
        assert self.coordinator is not None
        for encoded in self.args.fault:
            self.coordinator.inject_fault(*parse_fault_option(encoded))

    def start_order(self, preset: str = "A") -> None:
        assert self.coordinator is not None
        product = self.coordinator.start_order(preset, now=self.scene.time)
        self.scene.reset(product, raw=True)
        self.visualized_faults.clear()
        print(f"[ORDER] {product.order_id}: preset {preset}", flush=True)

    def start_segment(self, segment: str) -> None:
        """Reset to the deterministic prerequisites for one UI segment."""

        assert self.coordinator is not None
        if self.coordinator.product is not None:
            self.coordinator.reset()
        product = self.coordinator.start_segment(segment, now=self.scene.time)
        self.scene.reset(product, raw=True)
        if segment != "pick_place":
            self.scene.registry.place_base_on_tray(snap=True)
            for fin in product.active_fins:
                self.scene.registry.place_fin_in_slot(fin.fin_id, snap=True)
        if segment == "inspection_2":
            for path in product.active_paths:
                self.scene.registry.set_path_visible(path.path_id, True, coverage=1.0)
        self.visualized_faults.clear()
        print(f"[SEGMENT] {segment}", flush=True)

    def process_command(self, command: dict[str, Any]) -> None:
        assert self.coordinator is not None
        kind = str(command.get("type", ""))
        if kind == "order":
            self.start_order(str(command.get("preset", "A")))
        elif kind == "segment":
            self.start_segment(str(command.get("segment", "")))
        elif kind == "fault":
            fault = self.coordinator.inject_fault(
                str(command.get("fault_type", "")),
                str(command.get("target", "")),
                str(command.get("severity", "recoverable")),
            )
            print(f"[FAULT] armed {fault.fault_type} {fault.target} {fault.severity}", flush=True)
        elif kind == "stop":
            self.coordinator.pause(self.scene.time)
        elif kind == "continue":
            self.coordinator.resume(self.scene.time)
        elif kind == "reset":
            self.coordinator.reset()
            self.scene.reset("A", raw=True)
            self.visualized_faults.clear()
        elif kind == "quit":
            self.running = False

    def drain_commands(self) -> None:
        while True:
            try:
                command = self.shared.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self.process_command(command)
            except Exception as exc:
                self.shared.update(status=f"command failed: {exc}", last_error=str(exc))
                print(f"[COMMAND] {exc}", file=sys.stderr, flush=True)

    def sync_fault_visuals(self) -> None:
        assert self.coordinator is not None
        product = self.coordinator.product
        if product is None:
            return
        for index, fault in enumerate(self.coordinator.faults):
            if not fault.applied or index in self.visualized_faults:
                continue
            if fault.fault_type == "fin_pose":
                fin = next((item for item in product.active_fins if item.fin_id == fault.target), None)
                if fin is not None:
                    registry = self.scene.registry
                    pose = registry.free_body_pose(fin.fin_id)
                    registry.set_weld(f"{fin.fin_id}_fixture_weld", False)
                    displaced = Pose(
                        pose.position + np.asarray([0.0, fin.position_error_m, 0.0]),
                        pose.quaternion,
                    )
                    registry.set_free_body_pose(fin.fin_id, displaced, forward=True)
                    registry.set_weld(
                        f"{fin.fin_id}_fixture_weld",
                        True,
                        recompute=("assembly_tray", fin.fin_id),
                        forward=True,
                    )
            elif fault.fault_type == "brazing_gap":
                path = next((item for item in product.active_paths if item.path_id == fault.target), None)
                if path is not None:
                    self.scene.registry.set_path_visible(path.path_id, True, coverage=path.coverage_ratio)
            self.visualized_faults.add(index)

    def sync_furnace(self) -> None:
        assert self.coordinator is not None
        if self.coordinator.product is None:
            return
        fraction = self.coordinator.product.furnace.door_fraction
        self.scene.registry.set_furnace_door(
            fraction,
            teleport=bool(self.args.headless or self.args.fast),
        )

    def check_safety(self) -> None:
        assert self.coordinator is not None
        if self.args.fast or not self.coordinator.running:
            return
        contacts = [
            contact
            for contact in self.safety.unexpected(self.scene.data)
            if not self.expected_task_contact(contact)
        ]
        if not contacts:
            return
        first = contacts[0]
        message = (
            f"unexpected contact {first.body1}/{first.geom1} <-> "
            f"{first.body2}/{first.geom2} ({first.distance * 1000:.2f} mm)"
        )
        if self.coordinator.product is not None:
            self.coordinator.product.fail(message, self.scene.time)
        self.scene.stop(message)
        print(f"[SAFETY] {message}", file=sys.stderr, flush=True)

    def expected_task_contact(self, contact: Any) -> bool:
        """Allow only shallow contact required by the currently leased skill."""

        assert self.coordinator is not None
        task = self.coordinator.active_task
        if task is None:
            return False
        pair = {contact.body1, contact.body2}
        task_type = str(task.task_type)
        if task_type in {"INSERT_FIN", "ADJUST_FIN"}:
            fin_id = str(task.payload.get("fin_id", ""))
            intended_finger_contact = fin_id in pair and any(
                body == "arm1_parallel_gripper" or body.startswith("arm1_gripper_") for body in pair
            )
            if intended_finger_contact:
                # MuJoCo reports overlap against the thin fin's long face;
                # 12 mm corresponds to the two 6 mm finger half-widths and is
                # limited strictly to the currently commanded fin.
                return contact.distance >= -0.0125
        if contact.distance < -0.003:
            return False
        arm1_tool_contact = "arm1_tool_rack" in pair and bool(
            pair & {"arm1_parallel_gripper", "arm1_suction_tool"}
        )
        arm1_flange_contact = "arm1_fr3_link7" in pair and bool(
            pair & {"arm1_parallel_gripper", "arm1_suction_tool"}
        )
        arm2_tool_contact = "arm2_tool_rack" in pair and bool(
            pair & {"arm2_brazing_dispenser", "arm2_tray_transfer"}
        )
        arm2_flange_contact = "arm2_fr3_link7" in pair and bool(
            pair & {"arm2_brazing_dispenser", "arm2_tray_transfer"}
        )
        raw_rack_contact = "raw_material_rack" in pair and any(
            body == "base_plate" or body.startswith("fin_") for body in pair
        )
        if task_type == "LOAD_BASE":
            return (
                arm1_tool_contact
                or arm1_flange_contact
                or raw_rack_contact
                or pair
                in (
                    {"arm1_suction_tool", "base_plate"},
                    {"base_plate", "assembly_tray"},
                    {"base_plate", "assembly_fixture"},
                )
            )
        if task_type in {"INSERT_FIN", "ADJUST_FIN"}:
            fin_id = str(task.payload.get("fin_id", ""))
            gripper_contact = fin_id in pair and any(
                body == "arm1_parallel_gripper" or body.startswith("arm1_gripper_") for body in pair
            )
            return (
                arm1_tool_contact
                or arm1_flange_contact
                or raw_rack_contact
                or gripper_contact
                or pair == {fin_id, "base_plate"}
            )
        if task_type in {"APPLY_MATERIAL", "REAPPLY_MATERIAL"}:
            path_id = str(task.payload.get("path_id", ""))
            fin_id = path_id.rsplit("_", 1)[0]
            return (
                arm2_tool_contact
                or arm2_flange_contact
                or pair
                == {
                    "arm2_brazing_dispenser",
                    fin_id,
                }
            )
        if task_type == "LOCK_FIXTURE":
            return arm2_tool_contact or arm2_flange_contact
        if task_type in {"LOAD_FURNACE", "UNLOAD_FURNACE"}:
            return arm2_tool_contact or arm2_flange_contact or pair == {"arm2_tray_transfer", "assembly_tray"}
        return False

    def tick(self) -> None:
        assert self.coordinator is not None
        self.drain_commands()
        self.coordinator.tick(self.scene.time)
        self.sync_fault_visuals()
        self.sync_furnace()
        self.scene.step()
        self.check_safety()
        product = self.coordinator.product
        stage = "IDLE" if product is None else product.stage.value
        if stage != self.last_stage:
            print(f"[STAGE] {stage}", flush=True)
            self.last_stage = stage

    def publish(self, viewer_running: bool) -> None:
        assert self.coordinator is not None
        snapshot = self.coordinator.snapshot(self.scene.time)
        product = self.coordinator.product
        active_task = self.coordinator.active_task
        active_path = ""
        if active_task is not None and str(active_task.actor) == "arm2":
            active_path = str(active_task.payload.get("path_id", ""))
        active_paths = [] if product is None else product.active_paths
        snapshot["tools"] = {
            "arm1": self.scene.arm1_tools.state,
            "arm2": self.scene.tools.state,
        }
        snapshot["arm2_process"] = {
            "current_path": active_path,
            "completed_paths": sum(bool(path.applied) for path in active_paths),
            "total_paths": len(active_paths),
            "tray_carrying": bool(
                self.scene.data.eq_active[self.scene.registry.equality_id("arm2_tray_carry")]
            ),
        }
        self.shared.update(snapshot, viewer_running=viewer_running)

    def render_camera(self) -> None:
        assert self.coordinator is not None
        try:
            frame = self.scene.camera_rgb(
                int(self.args.camera_width),
                int(self.args.camera_height),
                "arm3_wrist_camera",
            )
            active = bool(
                self.coordinator.product is not None
                and self.coordinator.product.stage
                in {OrderStage.PRE_INSPECTION, OrderStage.MATERIAL_INSPECTION, OrderStage.POST_INSPECTION}
            )
            self.shared.update_camera(
                rgb_frame_to_ppm(frame),
                width=frame.shape[1],
                height=frame.shape[0],
                active=active,
                status="inspection active" if active else "camera ready",
            )
        except Exception as exc:
            self.shared.update(camera_status=f"camera unavailable: {exc}")

    def run_headless(self) -> int:
        assert self.coordinator is not None
        while self.running and self.scene.time <= float(self.args.max_sim_time):
            self.tick()
            wall = time.monotonic()
            if wall - self.last_publish_wall >= 0.05:
                self.publish(False)
                self.last_publish_wall = wall
            if self.coordinator.terminal:
                break
        if not self.coordinator.terminal and self.coordinator.product is not None:
            self.coordinator.product.fail("headless simulation timeout", self.scene.time)
        self.publish(False)
        snapshot = self.coordinator.snapshot(self.scene.time)
        print(json.dumps(snapshot, ensure_ascii=False, indent=2), flush=True)
        if self.coordinator.product is None:
            return 0
        return 2 if self.coordinator.product.stage in {OrderStage.ERROR, OrderStage.MANUAL_REVIEW} else 0

    def run_viewer(self) -> int:
        import mujoco
        import mujoco.viewer

        assert self.coordinator is not None
        with mujoco.viewer.launch_passive(
            self.scene.model,
            self.scene.data,
            show_left_ui=bool(self.args.show_mujoco_ui),
            show_right_ui=bool(self.args.show_mujoco_ui),
        ) as viewer:
            viewer.cam.lookat[:] = [-0.05, 0.42, 0.25]
            viewer.cam.distance = 2.65
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -32
            self.shared.update(viewer_running=True)
            while viewer.is_running() and self.running:
                started = time.monotonic()
                self.tick()
                viewer.sync()
                wall = time.monotonic()
                if wall - self.last_publish_wall >= 0.08:
                    self.publish(True)
                    self.last_publish_wall = wall
                if wall - self.last_camera_wall >= 1.0 / float(self.args.camera_fps):
                    self.render_camera()
                    self.last_camera_wall = wall
                delay = float(self.args.dt) - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
        self.shared.update(viewer_running=False)
        return 0

    def close(self) -> None:
        self.shared.update(status="shutdown", viewer_running=False, camera_active=False)
        self.server.shutdown()
        self.server.server_close()
        if self.ui_process is not None and self.ui_process.poll() is None:
            self.ui_process.terminate()
        self.scene.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ui_client:
        return run_ui_client(args)
    application = BrazingApplication(args)
    try:
        application.start_services()
        application.arm_faults()
        if args.order:
            application.start_order(args.order)
        return application.run_headless() if args.headless else application.run_viewer()
    finally:
        application.close()


if __name__ == "__main__":
    raise SystemExit(main())
