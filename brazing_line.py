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
from brazing_sim.api import (
    SharedState,
    jsonable,
    rgb_frame_to_ppm,
    start_http_server,
    start_terminal_thread,
)
from brazing_sim.batch import BatchCoordinator
from brazing_sim.domain import BatchStage, OrderStage
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
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--order", choices=("A", "B", "C"), default=None)
    run_mode.add_argument(
        "--batch",
        choices=("A",),
        default=None,
        help="run the three-layer batch",
    )
    run_mode.add_argument("--order-file", default=None, help="run one strict flexible-order YAML")
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
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=5.0,
        help="Arm3 camera FPS while inspection is active (standby is capped at 2 FPS)",
    )
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


class SimulationRate:
    """Convert one viewer refresh into stable fixed-timestep simulation steps."""

    MINIMUM = 0.25
    MAXIMUM = 32.0

    def __init__(self) -> None:
        self.multiplier = 1.0
        self._step_credit = 0.0

    def adjust(self, action: str) -> float:
        if action == "accelerate":
            self.multiplier = min(self.MAXIMUM, self.multiplier * 2.0)
        elif action == "decelerate":
            self.multiplier = max(self.MINIMUM, self.multiplier / 2.0)
        else:
            raise ValueError("speed action must be accelerate or decelerate")
        return self.multiplier

    def steps_for_frame(self) -> int:
        """Return an integer step count while preserving fractional rates."""

        self._step_credit += self.multiplier
        steps = int(self._step_credit + 1.0e-12)
        self._step_credit -= steps
        return steps


class ViewerRenderScheduler:
    """Protect interactive viewer frames from the secondary camera renderer.

    MuJoCo's macOS renderer performs the main viewer and the Arm3 offscreen
    camera on the same graphics thread.  Pausing the secondary renderer while
    the user is orbiting/panning removes the periodic drag hitch without
    lowering the main-view model quality.
    """

    def __init__(
        self,
        camera_fps: float,
        *,
        standby_fps: float = 2.0,
        interaction_cooldown_s: float = 0.35,
    ) -> None:
        if camera_fps <= 0 or standby_fps <= 0 or interaction_cooldown_s < 0:
            raise ValueError("camera render rates must be positive")
        self.camera_fps = float(camera_fps)
        self.standby_fps = min(float(standby_fps), self.camera_fps)
        self.interaction_cooldown_s = float(interaction_cooldown_s)
        self.last_camera_render_wall = float("-inf")
        self.last_view_change_wall = float("-inf")
        self._view_signature: tuple[float, ...] | None = None

    @staticmethod
    def _signature(camera: Any) -> tuple[float, ...]:
        lookat = tuple(float(value) for value in camera.lookat)
        return (
            float(camera.azimuth),
            float(camera.elevation),
            float(camera.distance),
            *lookat,
        )

    def observe_view(self, camera: Any, now: float) -> bool:
        signature = self._signature(camera)
        moved = self._view_signature is not None and any(
            abs(current - previous) > 1.0e-5 for current, previous in zip(signature, self._view_signature)
        )
        self._view_signature = signature
        if moved:
            self.last_view_change_wall = float(now)
        return moved

    def camera_due(self, now: float, *, inspection_active: bool) -> bool:
        fps = self.camera_fps if inspection_active else self.standby_fps
        return bool(
            float(now) - self.last_camera_render_wall >= 1.0 / fps
            and float(now) - self.last_view_change_wall >= self.interaction_cooldown_s
        )

    def mark_camera_rendered(self, now: float) -> None:
        self.last_camera_render_wall = float(now)


class BrazingApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process_plan = None
        if args.order_file:
            from brazing_sim.flexible import build_process_plan

            self.process_plan = build_process_plan(args.order_file)
        motion = MotionConfig(dt=float(args.dt))
        initial_order = (
            self.process_plan.execution_spec
            if self.process_plan is not None
            else args.order or args.batch or "A"
        )
        self.scene = BrazingScene(args.xml, order=initial_order, motion_config=motion, raw=True)
        self.scene.model.opt.timestep = float(args.dt)
        self.coordinator: ProcessCoordinator | None = None
        self.batch_coordinator: BatchCoordinator | None = None
        actors = build_scene_actors(
            self.scene,
            self.current_product,
            fast=bool(args.fast),
        )
        self.coordinator = ProcessCoordinator(actors=actors, fast=bool(args.fast))
        self.batch_coordinator = BatchCoordinator(
            self.scene,
            self.coordinator,
            fast=bool(args.fast),
        )
        self.shared = SharedState()
        self.server = start_http_server(self.shared, args.host, int(args.port))
        self.actual_port = int(self.server.server_address[1])
        self.safety = ContactMonitor(self.scene.model)
        self.ui_process: subprocess.Popen[Any] | None = None
        self.visualized_faults: set[int] = set()
        self.last_stage = ""
        self.last_publish_wall = 0.0
        self.render_scheduler = ViewerRenderScheduler(float(args.camera_fps))
        self.simulation_rate = SimulationRate()
        self.command_error = ""
        self.running = True
        if self.process_plan is not None:
            from brazing_sim.flexible import validate_process_plan

            validate_process_plan(self.process_plan, self.scene)

    def batch_active(self) -> bool:
        return bool(self.batch_coordinator is not None and self.batch_coordinator.batch is not None)

    def current_product(self) -> Any:
        if self.batch_active():
            assert self.batch_coordinator is not None
            return self.batch_coordinator.product
        return None if self.coordinator is None else self.coordinator.product

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
        assert self.coordinator is not None and self.batch_coordinator is not None
        if self.batch_active():
            self.batch_coordinator.reset()
        product = self.coordinator.start_order(preset, now=self.scene.time)
        self.scene.reset(product, raw=True)
        self.visualized_faults.clear()
        print(f"[ORDER] {product.order_id}: preset {preset}", flush=True)

    def start_segment(self, segment: str) -> None:
        """Reset to the deterministic prerequisites for one UI segment."""

        assert self.coordinator is not None and self.batch_coordinator is not None
        if segment == "rack_transfer":
            if self.batch_active():
                self.batch_coordinator.reset()
            self.batch_coordinator.start_transfer_demo(now=self.scene.time)
            self.visualized_faults.clear()
            print("[SEGMENT] rack_transfer", flush=True)
            return
        if self.batch_active():
            self.batch_coordinator.reset()
        if self.coordinator.product is not None:
            self.coordinator.reset()
        product = self.coordinator.start_segment(segment, now=self.scene.time)
        self.scene.reset(product, raw=True)
        if segment != "pick_place":
            self.scene.registry.place_base_on_tray(snap=True)
        if segment == "arm2_motion":
            # The segmented demonstration begins immediately after Arm1 has
            # placed the base, so reproduce its real predecessor tool state.
            self.scene.arm1_tools.change_tool("suction_tool")
        if segment in {"fin_assembly", "inspection_2", "furnace_cycle"}:
            self.scene.fixture_controller.configure_product(product.spec, product.fixture)
        if segment in {"inspection_2", "furnace_cycle"}:
            for fin in product.active_fins:
                self.scene.registry.place_fin_in_slot(fin.fin_id, snap=True)
        if segment in {"inspection_1", "fin_assembly", "inspection_2", "furnace_cycle"}:
            for path in product.active_paths:
                self.scene.registry.set_path_visible(path.path_id, True, coverage=1.0)
        self.visualized_faults.clear()
        print(f"[SEGMENT] {segment}", flush=True)

    def start_batch(self, preset: str = "A", layers: int = 3) -> None:
        assert self.coordinator is not None and self.batch_coordinator is not None
        if self.batch_active():
            self.batch_coordinator.reset()
        batch = self.batch_coordinator.start_batch(
            preset,
            layers=layers,
            now=self.scene.time,
        )
        self.visualized_faults.clear()
        print(f"[BATCH] {batch.batch_id}: {layers} x preset {preset}", flush=True)

    def start_flexible_order(self) -> None:
        """Start all units from the prevalidated YAML ProcessPlan."""

        assert self.process_plan is not None and self.batch_coordinator is not None
        if self.batch_active():
            self.batch_coordinator.reset()
        batch = self.batch_coordinator.start_process_plan(
            self.process_plan,
            now=self.scene.time,
        )
        self.visualized_faults.clear()
        summary = self.process_plan.summary()
        print(
            f"[PLAN] {batch.batch_id}: {summary['quantity']}件 {summary['preset']}型，"
            f"{summary['fin_count']}片/{summary['path_count']}路径，"
            f"层位{summary['rack_layers']}",
            flush=True,
        )

    def process_command(self, command: dict[str, Any]) -> None:
        assert self.coordinator is not None
        kind = str(command.get("type", ""))
        if kind == "order":
            self.start_order(str(command.get("preset", "A")))
        elif kind == "batch":
            self.start_batch(
                str(command.get("preset", "A")),
                int(command.get("layers", 3)),
            )
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
            if self.batch_active():
                assert self.batch_coordinator is not None
                self.batch_coordinator.pause(self.scene.time)
            else:
                self.coordinator.pause(self.scene.time)
        elif kind == "continue":
            if self.batch_active():
                assert self.batch_coordinator is not None
                self.batch_coordinator.resume(self.scene.time)
            else:
                self.coordinator.resume(self.scene.time)
        elif kind == "speed":
            multiplier = self.simulation_rate.adjust(str(command.get("action", "")))
            print(f"[SPEED] {multiplier:g}x", flush=True)
        elif kind == "reset":
            if self.batch_active():
                assert self.batch_coordinator is not None
                self.batch_coordinator.reset()
            else:
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
                self.command_error = ""
                self.shared.update(
                    status=f"command accepted: {command.get('type', '')}",
                    last_error="",
                )
            except Exception as exc:
                self.command_error = str(exc)
                self.shared.update(status=f"command failed: {exc}", last_error=str(exc))
                print(f"[COMMAND] {exc}", file=sys.stderr, flush=True)

    def sync_fault_visuals(self) -> None:
        assert self.coordinator is not None
        product = self.current_product()
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
        if self.batch_active() and self.batch_coordinator is not None:
            if self.batch_coordinator.batch is None:
                return
            fraction = self.batch_coordinator.batch.furnace.door_fraction
        elif self.coordinator.product is not None:
            fraction = self.coordinator.product.furnace.door_fraction
        else:
            return
        self.scene.registry.set_furnace_door(
            fraction,
            teleport=bool(self.args.headless or self.args.fast),
        )

    def check_safety(self) -> None:
        assert self.coordinator is not None
        batch_running = bool(self.batch_coordinator is not None and self.batch_coordinator.running)
        active_controller_running = batch_running if self.batch_active() else self.coordinator.running
        if self.args.fast or not active_controller_running:
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
        if self.batch_active() and self.batch_coordinator is not None:
            if self.batch_coordinator.batch is not None:
                self.batch_coordinator.batch.fail(message, self.scene.time)
        elif self.coordinator.product is not None:
            self.coordinator.product.fail(message, self.scene.time)
        self.scene.stop(message)
        print(f"[SAFETY] {message}", file=sys.stderr, flush=True)

    def expected_task_contact(self, contact: Any) -> bool:
        """Allow only shallow contact required by the currently leased skill."""

        assert self.coordinator is not None
        pair = {contact.body1, contact.body2}
        product = self.current_product()
        contacted_fin = next((body for body in pair if body.startswith("fin_")), "")
        installed_fin = bool(
            product is not None
            and any(fin.fin_id == contacted_fin and fin.inserted for fin in product.active_fins)
        )
        press_contact = "fixture_upper_plate" in pair and installed_fin
        if (
            product is not None
            and product.fixture.press_force_held
            and product.stage
            in {
                OrderStage.FIXTURE_PRESSING,
                OrderStage.FIXTURE_LOCKING,
                OrderStage.READY_FOR_TRANSFER,
                OrderStage.FURNACE_LOADING,
                OrderStage.BRAZING,
                OrderStage.UNLOADING,
                OrderStage.POST_INSPECTION,
            }
            and press_contact
            and contact.distance >= -0.003
        ):
            # The two retained transverse bars remain in controlled contact
            # throughout conveyor transport and the furnace dwell.
            return True
        if (
            product is not None
            and product.stage is OrderStage.FIN_ASSEMBLY
            and installed_fin
            and contact.distance >= -0.0015
            and any(body.startswith("arm1_gripper_") for body in pair)
        ):
            # At the exact task boundary MuJoCo can retain one shallow contact
            # from the just-opened fingers for a single step, after the fin
            # task has already released its lease.  It is the intended grasp
            # interface, not a cross-arm collision; deeper overlap is still a
            # hard safety stop.
            return True
        background_arm1 = self.coordinator.background_tasks.get("arm1")
        if (
            background_arm1 is not None
            and str(background_arm1.task_type) == "PREPARE_FIN_TOOL"
            and contact.distance >= -0.003
        ):
            arm1_tool_contact = "arm1_tool_rack" in pair and bool(
                pair & {"arm1_parallel_gripper", "arm1_suction_tool"}
            )
            arm1_flange_contact = "arm1_fr3_link7" in pair and bool(
                pair & {"arm1_parallel_gripper", "arm1_suction_tool"}
            )
            if arm1_tool_contact or arm1_flange_contact:
                return True
        task = self.coordinator.active_task
        if task is None:
            return False
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
        arm2_flange_contact = "arm2_fr3_link7" in pair and bool(pair & {"arm2_dual_brazing_dispenser_tool"})
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
            return arm2_flange_contact or pair == {"arm2_dual_brazing_dispenser_tool", "base_plate"}
        if task_type == "PRESS_FIXTURE":
            return "fixture_upper_plate" in pair and any(body.startswith("fin_") for body in pair)
        return False

    def tick(self) -> None:
        assert self.coordinator is not None and self.batch_coordinator is not None
        self.drain_commands()
        if self.batch_active():
            self.batch_coordinator.tick(self.scene.time)
        else:
            self.coordinator.tick(self.scene.time)
        self.sync_fault_visuals()
        self.sync_furnace()
        self.scene.step()
        self.check_safety()
        product = self.current_product()
        stage = (
            self.batch_coordinator.batch.stage.value
            if self.batch_active() and self.batch_coordinator.batch is not None
            else "IDLE" if product is None else product.stage.value
        )
        if stage != self.last_stage:
            print(f"[STAGE] {stage}", flush=True)
            self.last_stage = stage

    def publish(self, viewer_running: bool) -> None:
        assert self.coordinator is not None
        snapshot = (
            self.batch_coordinator.snapshot(self.scene.time)
            if self.batch_active() and self.batch_coordinator is not None
            else self.coordinator.snapshot(self.scene.time)
        )
        product = self.current_product()
        active_task = (
            self.batch_coordinator.active_task
            if self.batch_active() and self.batch_coordinator is not None
            else self.coordinator.active_task
        )
        active_path = ""
        if active_task is not None and str(active_task.actor) == "arm2":
            active_path = str(active_task.payload.get("path_id", ""))
            if not active_path:
                active_path = ",".join(str(value) for value in active_task.payload.get("path_ids", ()))
        active_paths = [] if product is None else product.active_paths
        snapshot["tools"] = {
            "arm1": self.scene.arm1_tools.state,
            "arm2": self.scene.tools.state,
        }
        snapshot["preflight"] = self.scene.preflight_report.as_dict()
        snapshot["simulation_speed"] = self.simulation_rate.multiplier
        if self.command_error:
            snapshot["last_error"] = self.command_error
        snapshot["arm2_process"] = {
            "current_path": active_path,
            "completed_paths": sum(bool(path.applied) for path in active_paths),
            "total_paths": len(active_paths),
        }
        if not self.batch_active():
            snapshot.update(
                batch={},
                rack={"shelves": []},
                transfer={
                    "phase": "IDLE",
                    "step": "",
                    "unit_id": None,
                    "shelf_index": None,
                    "lift_height_m": 0.0,
                    "outfeed_position_m": 0.0,
                    "pusher_position_m": 0.0,
                    "pusher_extension_ratio": 0.0,
                    "lock_position_m": 0.0,
                    "output_position_m": 0.0,
                    "moving": False,
                    "prefetch_unit_index": None,
                    "prefetch_complete_index": None,
                    "parallel_axes": [],
                    "parallel_active": False,
                },
            )
        self.shared.update(snapshot, viewer_running=viewer_running)

    def render_camera(self) -> None:
        assert self.coordinator is not None
        try:
            product = self.current_product()
            frame = self.scene.camera_rgb(
                int(self.args.camera_width),
                int(self.args.camera_height),
                "arm3_wrist_camera",
            )
            active = bool(
                product is not None
                and product.stage
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

    def camera_inspection_active(self) -> bool:
        product = self.current_product()
        return bool(
            product is not None
            and product.stage
            in {
                OrderStage.PRE_INSPECTION,
                OrderStage.MATERIAL_INSPECTION,
                OrderStage.POST_INSPECTION,
            }
        )

    def advance_simulation_frame(self) -> int:
        """Advance the real coordinator and MuJoCo model by one speed budget."""

        steps = self.simulation_rate.steps_for_frame()
        for _ in range(steps):
            self.tick()
        return steps

    def run_headless(self) -> int:
        assert self.coordinator is not None and self.batch_coordinator is not None
        while self.running and self.scene.time <= float(self.args.max_sim_time):
            # Keep API/terminal speed commands semantically identical across
            # viewer and headless modes. Headless remains unthrottled, but a
            # 32x budget still means 32 real coordinator + mj_step updates per
            # outer loop rather than a cosmetic state-label change.
            self.drain_commands()
            self.advance_simulation_frame()
            wall = time.monotonic()
            if wall - self.last_publish_wall >= 0.05:
                self.publish(False)
                self.last_publish_wall = wall
            terminal = self.batch_coordinator.terminal if self.batch_active() else self.coordinator.terminal
            if terminal:
                break
        terminal = self.batch_coordinator.terminal if self.batch_active() else self.coordinator.terminal
        if not terminal and self.current_product() is not None:
            if self.batch_active() and self.batch_coordinator.batch is not None:
                self.batch_coordinator.batch.fail("headless simulation timeout", self.scene.time)
            else:
                self.current_product().fail("headless simulation timeout", self.scene.time)
        self.publish(False)
        snapshot = (
            self.batch_coordinator.snapshot(self.scene.time)
            if self.batch_active()
            else self.coordinator.snapshot(self.scene.time)
        )
        print(json.dumps(jsonable(snapshot), ensure_ascii=False, indent=2), flush=True)
        if self.current_product() is None:
            return 0
        if self.batch_active() and self.batch_coordinator.batch is not None:
            return (
                2 if self.batch_coordinator.batch.stage in {BatchStage.ERROR, BatchStage.MANUAL_REVIEW} else 0
            )
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
            viewer.cam.lookat[:] = [-0.05, 0.62, 0.25]
            viewer.cam.distance = 2.90
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -32
            self.render_scheduler.observe_view(viewer.cam, time.monotonic())
            transfer_demo_focused = False
            self.shared.update(viewer_running=True)
            while viewer.is_running() and self.running:
                started = time.monotonic()
                # Commands are consumed before the frame budget is calculated,
                # so a speed button affects the process without restarting it.
                self.drain_commands()
                if not self.running:
                    break
                transfer_demo = bool(
                    self.batch_coordinator is not None and self.batch_coordinator.transfer_demo
                )
                if transfer_demo and not transfer_demo_focused:
                    # The standalone rack-transfer segment is an equipment
                    # demonstration, so frame the lift, telescopic fork and
                    # three shelves once.  The user remains free to orbit or
                    # zoom immediately afterwards.
                    viewer.cam.lookat[:] = [0.0, 0.91, 0.36]
                    viewer.cam.distance = 1.55
                    viewer.cam.azimuth = 145
                    viewer.cam.elevation = -22
                    transfer_demo_focused = True
                elif not transfer_demo:
                    transfer_demo_focused = False
                self.advance_simulation_frame()
                viewer.sync()
                wall = time.monotonic()
                if wall - self.last_publish_wall >= 0.08:
                    self.publish(True)
                    self.last_publish_wall = wall
                self.render_scheduler.observe_view(viewer.cam, wall)
                if self.render_scheduler.camera_due(
                    wall,
                    inspection_active=self.camera_inspection_active(),
                ):
                    # Mark before rendering so a transient camera failure does
                    # not trigger an expensive retry on every viewer frame.
                    self.render_scheduler.mark_camera_rendered(wall)
                    self.render_camera()
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
        if args.batch:
            application.start_batch(args.batch)
        elif args.order_file:
            application.start_flexible_order()
        elif args.order:
            application.start_order(args.order)
        return application.run_headless() if args.headless else application.run_viewer()
    finally:
        application.close()


if __name__ == "__main__":
    raise SystemExit(main())
