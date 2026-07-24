"""Three-FR3 heat-sink brazing-line simulation entry point."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

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
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
from brazing_sim.experiments import MetricsCollector
from brazing_sim.execution import (
    build_async_line_skill_registry,
    build_physical_async_line_skill_registry,
)
from brazing_sim.fault_catalog import MANUAL_FAULT_CATALOG
from brazing_sim.fault_visuals import PhysicalFaultVisualizer
from brazing_sim.physical_task_projection import PhysicalTaskStatusProjector
from brazing_sim.recovery import FaultType as V2FaultType
from brazing_sim.process import ProcessCoordinator
from brazing_sim.safety import ContactMonitor
from brazing_sim.scene import BrazingScene
from brazing_sim.async_line_router import AsyncLineProcessRouter
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
    run_mode.add_argument("--orders-file", default=None, help="run a V2 multi-order YAML queue")
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


class RealTimeFactorTracker:
    """Measure achieved simulation-time progress against wall-clock time."""

    def __init__(self, *, sample_window_s: float = 0.25, smoothing: float = 0.35) -> None:
        if sample_window_s <= 0.0 or not 0.0 < smoothing <= 1.0:
            raise ValueError("RTF tracker window and smoothing must be positive")
        self.sample_window_s = float(sample_window_s)
        self.smoothing = float(smoothing)
        self.actual_rtf = 0.0
        self._last_sim_time: float | None = None
        self._last_wall_time: float | None = None
        self._sim_accumulator = 0.0
        self._wall_accumulator = 0.0
        self._has_sample = False

    def reset(self, sim_time: float | None = None, wall_time: float | None = None) -> None:
        self.actual_rtf = 0.0
        self._last_sim_time = None if sim_time is None else float(sim_time)
        self._last_wall_time = None if wall_time is None else float(wall_time)
        self._sim_accumulator = 0.0
        self._wall_accumulator = 0.0
        self._has_sample = False

    def observe(self, sim_time: float, wall_time: float | None = None) -> float:
        wall = time.monotonic() if wall_time is None else float(wall_time)
        simulation = float(sim_time)
        if self._last_sim_time is None or self._last_wall_time is None:
            self._last_sim_time = simulation
            self._last_wall_time = wall
            return self.actual_rtf
        if simulation < self._last_sim_time or wall <= self._last_wall_time:
            self.reset(simulation, wall)
            return self.actual_rtf
        self._sim_accumulator += simulation - self._last_sim_time
        self._wall_accumulator += wall - self._last_wall_time
        self._last_sim_time = simulation
        self._last_wall_time = wall
        if self._wall_accumulator + 1.0e-12 < self.sample_window_s:
            return self.actual_rtf
        measured = self._sim_accumulator / max(self._wall_accumulator, 1.0e-12)
        self.actual_rtf = (
            measured
            if not self._has_sample
            else (1.0 - self.smoothing) * self.actual_rtf + self.smoothing * measured
        )
        self._has_sample = True
        self._sim_accumulator = 0.0
        self._wall_accumulator = 0.0
        return self.actual_rtf


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

    def camera_due(
        self,
        now: float,
        *,
        inspection_active: bool,
        speed_multiplier: float = 1.0,
    ) -> bool:
        fps = self.camera_fps if inspection_active else self.standby_fps
        # The offscreen wrist camera shares the macOS graphics thread with the
        # interactive viewer. At high simulation rates it is deliberately
        # capped so physics throughput and mouse interaction remain responsive.
        if speed_multiplier >= 16.0:
            fps = min(fps, 1.0)
        elif speed_multiplier >= 8.0:
            fps = min(fps, 2.0)
        return bool(
            float(now) - self.last_camera_render_wall >= 1.0 / fps
            and float(now) - self.last_view_change_wall >= self.interaction_cooldown_s
        )

    def mark_camera_rendered(self, now: float) -> None:
        self.last_camera_render_wall = float(now)


class ViewerFramePacer:
    """Decouple expensive OpenGL synchronization from fixed-step physics.

    A target speed of 32x means 32 control/physics ticks per application
    budget, not 32 full OpenGL redraws.  Keeping the viewer at a stable visual
    frame rate preserves model quality and mouse responsiveness while allowing
    the simulator to spend the remaining wall time advancing MuJoCo.
    """

    def __init__(self, fps: float = 30.0) -> None:
        if fps <= 0.0:
            raise ValueError("viewer fps must be positive")
        self.interval_s = 1.0 / float(fps)
        self.last_sync_wall = float("-inf")

    def due(self, now: float) -> bool:
        return float(now) - self.last_sync_wall + 1.0e-12 >= self.interval_s

    def set_fps(self, fps: float) -> None:
        if fps <= 0.0:
            raise ValueError("viewer fps must be positive")
        self.interval_s = 1.0 / float(fps)

    def mark_synced(self, now: float) -> None:
        self.last_sync_wall = float(now)


class BrazingApplication:
    V2_TICK_INTERVAL_S = 0.02
    V2_SNAPSHOT_INTERVAL_WALL_S = 0.25
    FAULT_VISUAL_INTERVAL_WALL_S = 1.0 / 30.0
    VIEWER_SYNC_FPS = 30.0

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process_plan = None
        self.process_plans: tuple[Any, ...] = ()
        if args.order_file:
            from brazing_sim.flexible import build_process_plan

            self.process_plan = build_process_plan(args.order_file)
        elif args.orders_file:
            from brazing_sim.flexible import load_order_plans

            self.process_plans = load_order_plans(args.orders_file)
            self.process_plan = self.process_plans[0]
        motion = MotionConfig(dt=float(args.dt))
        initial_order = (
            self.process_plan.execution_spec
            if self.process_plan is not None
            else args.order or args.batch or "A"
        )
        self.scene = BrazingScene(args.xml, order=initial_order, motion_config=motion, raw=True)
        self.fault_visuals = PhysicalFaultVisualizer(self.scene)
        self.scene.model.opt.timestep = float(args.dt)
        self.coordinator: ProcessCoordinator | None = None
        self.batch_coordinator: BatchCoordinator | None = None
        actors = build_scene_actors(
            self.scene,
            self.current_product,
            fast=bool(args.fast),
        )
        self.coordinator = ProcessCoordinator(actors=actors, fast=bool(args.fast))
        self.async_line_router = AsyncLineProcessRouter(self.scene, fast=bool(args.fast))
        self.coordinator.stage_gate = self._async_line_stage_gate
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
        self.rtf_tracker = RealTimeFactorTracker()
        self.manufacturing_runtime = ManufacturingRuntime(
            scheduler_mode="dynamic",
            flexible_cell=True,
            skill_registry=build_async_line_skill_registry(),
            context=self,
        )
        self.physical_task_projector = PhysicalTaskStatusProjector()
        self.v2_metrics = MetricsCollector()
        self.manufacturing_runtime.events.subscribe(None, self.v2_metrics.handle_event)
        self.v2_pending_plans: list[tuple[bool, Any]] = []
        self.v2_active_order_id: str | None = None
        self.v2_pipeline_active = False
        self.v2_raw_kit_order_id: str | None = None
        self.v2_consumed_materials: dict[str, set[str]] = {}
        self._last_v2_tick_sim = float("-inf")
        self._last_v2_snapshot_wall = float("-inf")
        self._last_fault_visual_wall = float("-inf")
        self._cached_v2_snapshot: dict[str, Any] | None = None
        self._last_furnace_fraction: float | None = None
        self.v2_furnace_visual: dict[str, Any] = {
            "phase": "IDLE",
            "temperature_c": 25.0,
            "progress": 0.0,
        }
        self._physical_fault_sequence = 0
        self._physical_fault_holds: list[dict[str, Any]] = []
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

    def _reset_multirate_clocks(self) -> None:
        """Reset clocks that must not leak across orders or scene resets."""

        self._last_v2_tick_sim = float("-inf")
        self._last_v2_snapshot_wall = float("-inf")
        self._last_fault_visual_wall = float("-inf")
        self._cached_v2_snapshot = None
        self._last_furnace_fraction = None
        self.v2_furnace_visual = {
            "phase": "IDLE",
            "temperature_c": 25.0,
            "progress": 0.0,
        }
        self.rtf_tracker.reset(self.scene.time, time.monotonic())

    def _async_line_stage_gate(self, stage: OrderStage, now: float) -> bool:
        """Move the pallet to its dedicated station before trajectory generation."""

        if not self.async_line_router.enabled:
            return True
        product = self.coordinator.product if self.coordinator is not None else None
        if product is None:
            return True
        background_busy = bool(self.coordinator.background_tasks)
        safe = not background_busy
        return self.async_line_router.gate(
            stage,
            now,
            product_token=product.order_id,
            safe_to_transfer=safe,
        )

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
        self._leave_v2_pipeline()
        if self.batch_active():
            self.batch_coordinator.reset()
        product = self.coordinator.start_order(preset, now=self.scene.time)
        self.scene.reset(product, raw=True)
        self.async_line_router.reset()
        self.async_line_router.activate()
        self._physical_fault_holds.clear()
        self.fault_visuals.reset()
        self.visualized_faults.clear()
        print(f"[ORDER] {product.order_id}: preset {preset}", flush=True)
        from brazing_sim.flexible import build_preset_plan

        self.manufacturing_runtime.reset(self.scene.time)
        self.physical_task_projector.reset()
        self._reset_multirate_clocks()
        self.manufacturing_runtime.submit_plan(build_preset_plan(preset, quantity=1), now=self.scene.time)

    def start_segment(self, segment: str) -> None:
        """Reset to the deterministic prerequisites for one UI segment."""

        assert self.coordinator is not None and self.batch_coordinator is not None
        self._leave_v2_pipeline()
        self.manufacturing_runtime.reset(self.scene.time)
        self.physical_task_projector.reset()
        self._reset_multirate_clocks()
        self.v2_pending_plans.clear()
        self.v2_active_order_id = None
        self.async_line_router.deactivate()
        self.async_line_router.reset()
        self._physical_fault_holds.clear()
        self.fault_visuals.reset()
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
        segment_station = {
            "pick_place": "s1",
            "arm2_motion": "s2a",
            "inspection_1": "s2b",
            "fin_assembly": "s3",
            "inspection_2": "s3",
            "furnace_cycle": "s3",
        }.get(segment, "s1")
        self.scene.registry.dock_assembly_tray_to_station(segment_station, snap=True)
        if segment == "furnace_cycle":
            self.async_line_router.activate()
            self.async_line_router.station = "s3"
            self.async_line_router.product_token = product.order_id
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
        self._leave_v2_pipeline()
        if self.batch_active():
            self.batch_coordinator.reset()
        batch = self.batch_coordinator.start_batch(
            preset,
            layers=layers,
            now=self.scene.time,
        )
        self.async_line_router.reset()
        self.async_line_router.activate()
        self._physical_fault_holds.clear()
        self.fault_visuals.reset()
        self.visualized_faults.clear()
        print(f"[BATCH] {batch.batch_id}: {layers} x preset {preset}", flush=True)
        from brazing_sim.flexible import build_preset_plan

        self.manufacturing_runtime.reset(self.scene.time)
        self.physical_task_projector.reset()
        self._reset_multirate_clocks()
        self.manufacturing_runtime.submit_plan(
            build_preset_plan(preset, quantity=layers), now=self.scene.time
        )

    def start_flexible_order(self) -> None:
        """Start all units from the prevalidated YAML ProcessPlan."""

        assert self.process_plan is not None and self.batch_coordinator is not None
        self._leave_v2_pipeline()
        if self.batch_active():
            self.batch_coordinator.reset()
        batch = self.batch_coordinator.start_process_plan(
            self.process_plan,
            now=self.scene.time,
        )
        self.async_line_router.reset()
        self.async_line_router.activate()
        self._physical_fault_holds.clear()
        self.fault_visuals.reset()
        self.visualized_faults.clear()
        summary = self.process_plan.summary()
        print(
            f"[PLAN] {batch.batch_id}: {summary['quantity']}件 {summary['preset']}型，"
            f"{summary['fin_count']}片/{summary['path_count']}路径，"
            f"层位{summary['rack_layers']}",
            flush=True,
        )
        self.manufacturing_runtime.reset(self.scene.time)
        self.physical_task_projector.reset()
        self._reset_multirate_clocks()
        self.manufacturing_runtime.submit_plan(self.process_plan, now=self.scene.time)

    def enqueue_v2_plan(self, plan: Any, *, urgent: bool = False) -> None:
        """Queue an order on the physical multi-pallet production runtime.

        The six standalone process buttons deliberately keep using the mature
        single-product actors. Complete orders belong to this scheduler: up
        to three physical pallets can occupy S1/S2A/S2B/S3 and Arm1/Arm2/Arm3
        may therefore work on different orders at the same simulated instant.
        """

        was_terminal = self.manufacturing_runtime.terminal
        if not self.v2_pipeline_active:
            if self.batch_active() and self.batch_coordinator is not None:
                batch = self.batch_coordinator.batch
                if batch is not None and not batch.terminal:
                    raise RuntimeError("当前旧流程仍在运行，请先停止或等待完成后再启动异步流水")
                self.batch_coordinator.reset()
            if self.coordinator is not None and self.coordinator.running:
                raise RuntimeError("当前旧流程仍在运行，请先停止或等待完成后再启动异步流水")
            self.manufacturing_runtime.reset(self.scene.time)
            self.manufacturing_runtime.set_skill_registry(
                build_physical_async_line_skill_registry(fast=bool(self.args.fast))
            )
            self.scene.registry.reset_batch_cell()
            self.scene.registry.set_workcell_visible(False)
            self.async_line_router.deactivate()
            self.v2_pending_plans.clear()
            self.v2_active_order_id = None
            self.v2_raw_kit_order_id = None
            self.v2_consumed_materials.clear()
            self.physical_task_projector.reset()
            self.v2_pipeline_active = True
        self.manufacturing_runtime.submit_plan(plan, urgent=urgent, now=self.scene.time)
        if self.v2_raw_kit_order_id is None or was_terminal:
            self.prepare_v2_raw_kit(
                plan.order.order_id,
                f"{plan.order.order_id}_UNIT_01",
            )
        if self.v2_active_order_id is None:
            self.v2_active_order_id = plan.order.order_id
        print(
            f"[ASYNC ORDER] queued {plan.order.order_id}: {plan.quantity} unit(s)",
            flush=True,
        )

    def prepare_v2_raw_kit(self, order_id: str, unit_id: str) -> None:
        """Expose the correct physical base/fins for the next Arm1 task."""

        from brazing_sim.config import create_product_state

        entry = self.manufacturing_runtime.orders[str(order_id)]
        product = create_product_state(
            entry.plan.execution_spec,
            order_id=str(unit_id),
            created_at=self.scene.time,
        )
        self.scene.registry.configure_async_raw_kit(product)
        consumed = self.v2_consumed_materials.setdefault(str(order_id), set())
        for item_name in sorted(consumed):
            self.scene.registry.set_async_raw_item_visible(item_name, False)
        self.v2_raw_kit_order_id = str(order_id)

    def enqueue_verified_plan(self, plan: Any, *, urgent: bool = False) -> None:
        """Queue one UI order on the authoritative physical process chain.

        The former UI path used short approximate skills that only moved each
        arm toward a station and then toggled batch-tray geometry.  It could
        therefore report a task complete without reproducing suction,
        gripping, serpentine dispensing or fixture callbacks.  The verified
        queue deliberately reuses ``ProcessCoordinator`` and
        ``BatchCoordinator`` so an inserted order is physically identical to
        the corresponding standalone demonstrations.
        """

        if self.v2_pipeline_active:
            self._leave_v2_pipeline()
        if self.coordinator is not None and self.coordinator.running and not self.batch_active():
            raise RuntimeError("当前分段/单订单演示仍在运行，请完成或重置后再加入规划订单")

        # A completely delivered previous queue must not block a fresh click.
        # Clear its planning mirror only after the physical queue is empty;
        # this preserves all current-order task nodes while they are running.
        if (
            self.v2_active_order_id is None
            and not self.v2_pending_plans
            and self.batch_coordinator is not None
            and (self.batch_coordinator.batch is None or self.batch_coordinator.terminal)
        ):
            self.manufacturing_runtime.reset(self.scene.time)
            self.manufacturing_runtime.set_skill_registry(build_async_line_skill_registry())
            self.physical_task_projector.reset()
            self._reset_multirate_clocks()

        self.manufacturing_runtime.submit_plan(plan, urgent=urgent, now=self.scene.time)
        self.manufacturing_runtime.pause(self.scene.time)
        queued = (bool(urgent), plan)
        if urgent:
            # Never pre-empt a physical action; urgent work only moves ahead
            # of waiting normal orders.
            insert_at = next(
                (index for index, (is_urgent, _plan) in enumerate(self.v2_pending_plans) if not is_urgent),
                len(self.v2_pending_plans),
            )
            self.v2_pending_plans.insert(insert_at, queued)
        else:
            self.v2_pending_plans.append(queued)
        self._start_next_v2_physical()
        print(
            f"[VERIFIED ORDER] queued {plan.order.order_id}: {plan.quantity} unit(s)",
            flush=True,
        )

    def _leave_v2_pipeline(self) -> None:
        if not self.v2_pipeline_active:
            return
        self.manufacturing_runtime.reset(self.scene.time)
        self.manufacturing_runtime.set_skill_registry(build_async_line_skill_registry())
        self.scene.registry.reset_batch_cell()
        self.scene.registry.set_workcell_visible(True)
        self.v2_pipeline_active = False
        self.v2_active_order_id = None
        self.v2_pending_plans.clear()
        self.v2_raw_kit_order_id = None
        self.v2_consumed_materials.clear()

    def _start_next_v2_physical(self) -> None:
        if self.v2_pipeline_active:
            return
        if self.v2_active_order_id is not None or not self.v2_pending_plans:
            return
        assert self.batch_coordinator is not None
        urgent, plan = self.v2_pending_plans.pop(0)
        del urgent
        if self.batch_active():
            self.batch_coordinator.reset()
        self.process_plan = plan
        self.batch_coordinator.start_process_plan(plan, now=self.scene.time)
        self.async_line_router.reset()
        self.async_line_router.activate()
        self.v2_active_order_id = plan.order.order_id
        print(f"[V2 ORDER] physical execution started: {self.v2_active_order_id}", flush=True)

    def _advance_v2_physical_queue(self) -> None:
        if self.v2_pipeline_active:
            return
        if self.v2_active_order_id is None or self.batch_coordinator is None:
            return
        batch = self.batch_coordinator.batch
        if batch is None or not batch.terminal:
            return
        # Capture the terminal physical evidence before the next queued order
        # resets the reusable workcell.  The projector keeps those completed
        # nodes green while the following order starts from PENDING.
        physical = self.batch_coordinator.snapshot(self.scene.time)
        physical["tools"] = {
            "arm1": self.scene.arm1_tools.state,
            "arm2": self.scene.tools.state,
        }
        self.physical_task_projector.project(
            self.manufacturing_runtime.snapshot(self.scene.time)["tasks"],
            physical,
            active_order_id=self.v2_active_order_id,
        )
        print(f"[V2 ORDER] physical execution completed: {self.v2_active_order_id}", flush=True)
        self.v2_active_order_id = None
        self._start_next_v2_physical()

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
        elif kind == "manual_fault_inject":
            self.inject_manual_fault(command)
        elif kind == "order_insert":
            from brazing_sim.flexible import build_custom_plan, build_inline_plan

            if str(command.get("mode", "preset")) == "custom":
                plan = build_custom_plan(
                    order_id=str(command.get("order_id")),
                    quantity=int(command.get("quantity", 1)),
                    priority=int(command.get("priority", 10)),
                    due_time=command.get("due_time"),
                    preferred_rack_layer=command.get("preferred_rack_layer"),
                    product=dict(command.get("custom_product") or {}),
                    route_strategy=str(command.get("route_strategy", "STANDARD")),
                )
            else:
                plan = build_inline_plan(
                    preset=str(command.get("preset", "A")),
                    order_id=str(command.get("order_id")),
                    quantity=int(command.get("quantity", 1)),
                    priority=int(command.get("priority", 10)),
                    due_time=command.get("due_time"),
                    preferred_rack_layer=command.get("preferred_rack_layer"),
                    route_strategy=str(command.get("route_strategy", "STANDARD")),
                )
            # Complete UI orders use the same multi-pallet backend as
            # ``--orders``. The former verified queue paused this DAG and
            # handed one plan at a time to BatchCoordinator, making the task
            # graph look parallel while MuJoCo still ran serially.
            self.enqueue_v2_plan(plan, urgent=bool(command.get("urgent", False)))
        elif kind == "resource_fault":
            self.manufacturing_runtime.inject_fault(
                V2FaultType.ARM_UNAVAILABLE,
                source=str(command.get("resource_id", "")),
                now=self.scene.time,
                details={
                    "resource_id": str(command.get("resource_id", "")),
                    "duration": command.get("duration"),
                    "fault_code": command.get("fault_code"),
                },
            )
        elif kind == "resource_recover":
            resource_id = str(command.get("resource_id", ""))
            self.manufacturing_runtime.recover_resource(resource_id, self.scene.time)
            self._recover_physical_fault_holds(resource_id)
        elif kind == "scheduler_replan":
            self.manufacturing_runtime.replanner.replan_ready_set(
                self.manufacturing_runtime.graph,
                str(command.get("reason", "operator")),
                self.scene.time,
            )
        elif kind == "recovery_action":
            self.manufacturing_runtime.recovery.action(
                str(command.get("recovery_id", "")),
                str(command.get("action", "")),
                self.scene.time,
            )
        elif kind == "stop":
            if self.v2_pipeline_active:
                pass
            elif self.batch_active():
                assert self.batch_coordinator is not None
                self.batch_coordinator.pause(self.scene.time)
            else:
                self.coordinator.pause(self.scene.time)
            self.manufacturing_runtime.pause(self.scene.time)
        elif kind == "continue":
            if self.v2_pipeline_active:
                pass
            elif self.batch_active():
                assert self.batch_coordinator is not None
                self.batch_coordinator.resume(self.scene.time)
            else:
                self.coordinator.resume(self.scene.time)
            self.manufacturing_runtime.resume(self.scene.time)
            for hold in self._physical_fault_holds:
                if hold["status"] == "ACTIVE":
                    hold["status"] = "RECOVERED"
                    hold["recovered_at"] = self.scene.time
        elif kind == "speed":
            multiplier = self.simulation_rate.adjust(str(command.get("action", "")))
            self.rtf_tracker.reset(self.scene.time, time.monotonic())
            print(f"[SPEED] {multiplier:g}x", flush=True)
        elif kind == "reset":
            self._leave_v2_pipeline()
            if self.batch_active():
                assert self.batch_coordinator is not None
                self.batch_coordinator.reset()
            else:
                self.coordinator.reset()
            self.scene.reset("A", raw=True)
            self.visualized_faults.clear()
            self.manufacturing_runtime.reset(self.scene.time)
            self.physical_task_projector.reset()
            self._reset_multirate_clocks()
            self.v2_pending_plans.clear()
            self.v2_active_order_id = None
            self.v2_pipeline_active = False
            self.v2_raw_kit_order_id = None
            self.v2_consumed_materials.clear()
            self.async_line_router.deactivate()
            self.async_line_router.reset()
            self._physical_fault_holds.clear()
            self.fault_visuals.reset()
        elif kind == "quit":
            self.running = False

    def inject_manual_fault(self, command: dict[str, Any]) -> None:
        """Route one friendly UI request to physical truth and/or V2 recovery."""

        assert self.coordinator is not None
        fault_type = str(command["fault_type"]).upper()
        definition = MANUAL_FAULT_CATALOG[fault_type]
        target = str(command.get("target", ""))
        severity = str(command.get("severity", "recoverable"))
        recoverable = severity != "severe"
        duration = command.get("duration_s")
        auto_recover = bool(command.get("auto_recover", True))

        if definition.physical_fault is not None:
            physical_target = target
            if definition.physical_fault == "furnace_profile":
                physical_target = "furnace"
            fault = self.coordinator.inject_fault(
                definition.physical_fault,
                physical_target,
                severity,
            )
            print(
                f"[FAULT UI] physical {fault.fault_type} {fault.target} {fault.severity} armed",
                flush=True,
            )

        if definition.runtime_fault is not None:
            details: dict[str, Any] = {"manual": True, "label_zh": definition.label_zh}
            source = "operator_ui"
            if definition.target_kind == "fin":
                details["fin_id"] = target
            elif definition.target_kind == "path":
                details["path_ids"] = [target]
            elif definition.target_kind == "arm":
                source = target
                details["resource_id"] = target
                if auto_recover and duration is not None:
                    details["duration"] = float(duration)
            elif definition.target_kind == "rack_layer":
                details["layer_id"] = int(target)
            request = self.manufacturing_runtime.arm_manual_fault(
                definition.runtime_fault,
                target=target,
                source=source,
                now=self.scene.time,
                recoverable=recoverable,
                details=details,
            )
            if fault_type in {
                "ARM_UNAVAILABLE",
                "RACK_LAYER_UNAVAILABLE",
                "ELEVATOR_TIMEOUT",
                "FORK_TIMEOUT",
                "FURNACE_DOOR_INTERLOCK",
                "CONTACT_SAFETY_STOP",
                "TRAY_STATE_INCONSISTENT",
            }:
                self._physical_fault_sequence += 1
                self._physical_fault_holds.append(
                    {
                        "request_id": f"PHYSICAL_{self._physical_fault_sequence:04d}",
                        "runtime_request_id": request.request_id,
                        "fault_type": fault_type,
                        "target": target,
                        "label_zh": definition.label_zh,
                        "status": "ARMED",
                        "armed_at": self.scene.time,
                        "auto_recover": auto_recover and recoverable,
                        "duration_s": 3.0 if duration is None else float(duration),
                        "started_at": None,
                        "resume_at": None,
                        "recovered_at": None,
                        "visual_only": fault_type == "RACK_LAYER_UNAVAILABLE",
                        "visual_target": "",
                    }
                )
        print(f"[FAULT UI] {definition.label_zh}: target={target or '-'}", flush=True)

    def _physical_fault_ready(self, hold: dict[str, Any]) -> bool:
        fault_type = str(hold["fault_type"])
        active_task = (
            self.batch_coordinator.active_task
            if self.batch_active() and self.batch_coordinator is not None
            else None if self.coordinator is None else self.coordinator.active_task
        )
        if fault_type == "ARM_UNAVAILABLE":
            return active_task is not None and str(active_task.actor).upper() == str(hold["target"]).upper()
        if fault_type in {"CONTACT_SAFETY_STOP", "TRAY_STATE_INCONSISTENT"}:
            return active_task is not None or self.batch_active()
        if fault_type == "RACK_LAYER_UNAVAILABLE":
            return self.batch_active()
        if fault_type == "FURNACE_DOOR_INTERLOCK" and not self.batch_active():
            product = None if self.coordinator is None else self.coordinator.product
            return product is not None and product.stage in {
                OrderStage.FURNACE_LOADING,
                OrderStage.BRAZING,
            }
        if not self.batch_active() or self.batch_coordinator is None:
            return False
        phase = self.batch_coordinator.transfer_actor.phase.value
        if fault_type in {"ELEVATOR_TIMEOUT", "FORK_TIMEOUT"}:
            return phase in {"CONVEYING_IN", "CONVEYING_OUT"}
        if fault_type == "FURNACE_DOOR_INTERLOCK":
            batch = self.batch_coordinator.batch
            return batch is not None and batch.stage.value in {"READY_FOR_BRAZING", "BRAZING"}
        return False

    def _set_physical_process_paused(self, paused: bool) -> None:
        if self.batch_active() and self.batch_coordinator is not None:
            if paused:
                self.batch_coordinator.pause(self.scene.time)
            else:
                self.batch_coordinator.resume(self.scene.time)
        elif self.coordinator is not None and self.coordinator.product is not None:
            if paused:
                self.coordinator.pause(self.scene.time)
            else:
                self.coordinator.resume(self.scene.time)

    def _service_physical_fault_holds(self) -> None:
        active = next(
            (hold for hold in self._physical_fault_holds if hold["status"] == "ACTIVE"),
            None,
        )
        if active is not None:
            resume_at = active.get("resume_at")
            if resume_at is not None and self.scene.time >= float(resume_at):
                if not active.get("visual_only", False):
                    self._set_physical_process_paused(False)
                active["status"] = "RECOVERED"
                active["recovered_at"] = self.scene.time
                if active["fault_type"] == "FURNACE_DOOR_INTERLOCK":
                    self._last_furnace_fraction = None
                print(f"[FAULT UI] physical recovery complete: {active['label_zh']}", flush=True)
            return
        physical_terminal = (
            self.batch_coordinator.terminal
            if self.batch_active() and self.batch_coordinator is not None
            else bool(self.coordinator is not None and self.coordinator.terminal)
        )
        if physical_terminal:
            for hold in self._physical_fault_holds:
                if hold["status"] == "ARMED":
                    hold["status"] = "MISSED"
            return
        pending = next(
            (
                hold
                for hold in self._physical_fault_holds
                if hold["status"] == "ARMED" and self._physical_fault_ready(hold)
            ),
            None,
        )
        if pending is None:
            return
        active_task = (
            self.batch_coordinator.active_task
            if self.batch_active() and self.batch_coordinator is not None
            else None if self.coordinator is None else self.coordinator.active_task
        )
        pending["visual_target"] = "" if active_task is None else str(active_task.actor)
        if not pending.get("visual_only", False):
            self._set_physical_process_paused(True)
        pending["status"] = "ACTIVE"
        pending["started_at"] = self.scene.time
        pending["resume_at"] = (
            self.scene.time + float(pending["duration_s"]) if pending["auto_recover"] else None
        )
        if pending["fault_type"] in {"ELEVATOR_TIMEOUT", "FORK_TIMEOUT"}:
            position = self.scene.registry.batch_joint_position("batch_outfeed_joint")
            self.scene.registry.set_batch_joint_target(
                "batch_outfeed_joint", "batch_outfeed_actuator", position
            )
        print(f"[FAULT UI] physical hold active: {pending['label_zh']}", flush=True)

    def _recover_physical_fault_holds(self, resource_id: str = "") -> None:
        normalized = str(resource_id).upper()
        changed = False
        for hold in self._physical_fault_holds:
            if hold["status"] not in {"ARMED", "ACTIVE"}:
                continue
            if normalized and hold["fault_type"] == "ARM_UNAVAILABLE" and hold["target"] != normalized:
                continue
            hold["status"] = "RECOVERED"
            hold["recovered_at"] = self.scene.time
            if hold["fault_type"] == "FURNACE_DOOR_INTERLOCK":
                self._last_furnace_fraction = None
            changed = True
        if changed:
            self._set_physical_process_paused(False)

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
        wall = time.monotonic()
        if wall - self._last_fault_visual_wall < self.FAULT_VISUAL_INTERVAL_WALL_S:
            return
        self._last_fault_visual_wall = wall
        if self.v2_pipeline_active:
            return
        product = self.current_product()
        if product is None:
            return
        for index, fault in enumerate(self.coordinator.faults):
            if not fault.applied or index in self.visualized_faults:
                continue
            if fault.fault_type in {"fin_pose", "fin_pick", "fin_insert"}:
                fin = next((item for item in product.active_fins if item.fin_id == fault.target), None)
                if fin is not None:
                    registry = self.scene.registry
                    registry.set_weld(f"{fin.fin_id}_fixture_weld", False)
                    if fault.fault_type == "fin_pick":
                        raw_pose = registry.site_pose(f"raw_{fin.fin_id}_site")
                        registry.set_free_body_pose(fin.fin_id, raw_pose, forward=True)
                        registry.set_weld(
                            f"raw_{fin.fin_id}_rack_weld",
                            True,
                            recompute=("raw_material_rack", fin.fin_id),
                            forward=True,
                        )
                    else:
                        angle = np.deg2rad(float(fin.verticality_error_deg))
                        position = np.asarray(fin.actual_position, dtype=float).copy()
                        position[2] += float(fin.root_gap_m)
                        local_pose = Pose(
                            position,
                            np.asarray([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0]),
                        )
                        displaced = registry.product_pose().transformed(local_pose)
                        registry.set_free_body_pose(fin.fin_id, displaced, forward=True)
                        registry.set_weld(
                            f"{fin.fin_id}_fixture_weld",
                            True,
                            recompute=("assembly_tray", fin.fin_id),
                            forward=True,
                        )
            self.visualized_faults.add(index)
        self.fault_visuals.sync_quality(product)
        active_task = (
            self.batch_coordinator.active_task
            if self.batch_active() and self.batch_coordinator is not None
            else self.coordinator.active_task
        )
        self.fault_visuals.sync_equipment(
            self._physical_fault_holds,
            now=self.scene.time,
            active_actor="" if active_task is None else str(active_task.actor),
        )
        # Refresh mocap safety markers immediately for this viewer frame.
        self.scene.mujoco.mj_kinematics(self.scene.model, self.scene.data)

    def sync_furnace(self) -> None:
        assert self.coordinator is not None
        if self.v2_pipeline_active:
            return
        if self.batch_active() and self.batch_coordinator is not None:
            # BatchCoordinator already updates the door together with its
            # furnace interlocks. Re-applying it here used to perform the same
            # qpos/kinematics work twice per simulation step.
            return
        elif self.coordinator.product is not None:
            fraction = self.coordinator.product.furnace.door_fraction
        else:
            return
        if self._last_furnace_fraction is not None and abs(fraction - self._last_furnace_fraction) <= 1.0e-12:
            return
        self.scene.registry.set_furnace_door(
            fraction,
            teleport=bool(self.args.headless or self.args.fast),
        )
        self._last_furnace_fraction = fraction

    def check_safety(self) -> None:
        assert self.coordinator is not None
        batch_running = bool(self.batch_coordinator is not None and self.batch_coordinator.running)
        active_controller_running = (
            bool(self.manufacturing_runtime.executor.active)
            if self.v2_pipeline_active
            else batch_running if self.batch_active() else self.coordinator.running
        )
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

    def _tick_v2_runtime_if_due(self, *, poll_executor: bool = True) -> bool:
        """Run the planning scheduler at 50 Hz instead of every physics step."""

        now = float(self.scene.time)
        if now + 1.0e-12 < self._last_v2_tick_sim:
            self._last_v2_tick_sim = float("-inf")
        if now - self._last_v2_tick_sim + 1.0e-12 < self.V2_TICK_INTERVAL_S:
            return False
        if poll_executor:
            self.manufacturing_runtime.tick(now)
        else:
            self.manufacturing_runtime.tick(now, poll_executor=False)
        self._last_v2_tick_sim = now
        return True

    def _log_stage_change(self) -> None:
        product = self.current_product()
        if self.v2_pipeline_active:
            stage = "COMPLETE" if self.manufacturing_runtime.terminal else "PARALLEL_PRODUCTION"
        else:
            stage = (
                self.batch_coordinator.batch.stage.value
                if self.batch_active()
                and self.batch_coordinator is not None
                and self.batch_coordinator.batch is not None
                else "IDLE" if product is None else product.stage.value
            )
        if stage != self.last_stage:
            print(f"[STAGE] {stage}", flush=True)
            self.last_stage = stage

    def service_simulation_frame(self) -> None:
        """Service commands and visual bookkeeping outside the physics loop."""

        self.drain_commands()
        self._advance_v2_physical_queue()
        self.sync_fault_visuals()
        self._log_stage_change()

    def tick(self) -> None:
        """Advance one fixed MuJoCo/control step.

        Robot control and collision safety remain tied to every physics step.
        Task-DAG scheduling is intentionally rate-limited because it operates
        on events and simulated task durations rather than rigid-body dynamics.
        """

        assert self.coordinator is not None and self.batch_coordinator is not None
        self._service_physical_fault_holds()
        if self.v2_pipeline_active:
            # Running physical skills are interpolated on every MuJoCo step;
            # DAG scheduling and resource scoring remain at 50 Hz.
            self.manufacturing_runtime.advance_active_skills(self.scene.time)
            self._tick_v2_runtime_if_due(poll_executor=False)
        else:
            self.async_line_router.tick(self.scene.time)
            if self.batch_active():
                self.batch_coordinator.tick(self.scene.time)
            else:
                self.coordinator.tick(self.scene.time)
            self._tick_v2_runtime_if_due()
        self.sync_furnace()
        if any(
            hold.get("status") == "ACTIVE" and hold.get("fault_type") == "FURNACE_DOOR_INTERLOCK"
            for hold in self._physical_fault_holds
        ):
            # A real interlock fault leaves the door visibly stuck part-open;
            # recovery hands the actuator back to the furnace actor.
            self.scene.registry.set_furnace_door(0.35, teleport=False)
        self.scene.step()
        self.check_safety()

    @staticmethod
    def _active_task_context(active_task: Any) -> tuple[str, dict[str, Any]]:
        if active_task is None:
            return "", {}
        return str(active_task.task_type), dict(active_task.payload)

    def _build_v2_presentation(self, physical: dict[str, Any], active_task: Any) -> dict[str, Any]:
        """Combine planning metadata with physically authoritative statuses."""

        v2 = self.manufacturing_runtime.snapshot(self.scene.time)
        if self.v2_pipeline_active:
            tasks = [dict(item) for item in v2["tasks"]]
        else:
            active_type, active_payload = self._active_task_context(active_task)
            tasks = self.physical_task_projector.project(
                v2["tasks"],
                physical,
                active_task_type=active_type,
                active_task_payload=active_payload,
                active_order_id=self.v2_active_order_id,
            )
        scheduler = dict(v2["scheduler"])
        scheduler["ready_count"] = sum(task["status"] == "READY" for task in tasks)
        scheduler["running_count"] = sum(task["status"] == "RUNNING" for task in tasks)
        orders = [dict(item) for item in v2["orders"]]
        terminal = {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}
        tasks_by_order: dict[str, list[dict[str, Any]]] = {}
        for task in tasks:
            tasks_by_order.setdefault(str(task.get("order_id", "")), []).append(task)
        for order in orders:
            order_tasks = tasks_by_order.get(str(order.get("order_id", "")), [])
            completed = sum(task.get("status") in terminal for task in order_tasks)
            order["progress"] = 0.0 if not order_tasks else completed / len(order_tasks)
            runtime_status = str(order.get("status", ""))
            # A recovered task deliberately remains FAILED in the immutable
            # event history while its recovery branch succeeds.  In that
            # case the order runtime is the authoritative terminal result;
            # do not repaint a genuinely completed order as RUNNING merely
            # because the original failed node is still visible in the DAG.
            if runtime_status in {"COMPLETED", "MANUAL_REVIEW", "CANCELLED"}:
                order["status"] = runtime_status
            elif order_tasks and all(
                task.get("status") in {"SUCCEEDED", "CANCELLED"} for task in order_tasks
            ):
                order["status"] = "COMPLETED"
            elif str(order.get("order_id", "")) == str(self.v2_active_order_id or ""):
                order["status"] = "RUNNING"
            elif any(
                queued_plan.order.order_id == str(order.get("order_id", ""))
                for _urgent, queued_plan in self.v2_pending_plans
            ):
                order["status"] = "QUEUED"
        physical_line = self.scene.registry.async_line_snapshot()
        async_line = dict(v2.get("async_line", {}))
        async_line.update(physical_line)
        verified_queue = bool(self.v2_active_order_id or self.v2_pending_plans)
        if self.v2_pipeline_active:
            async_line["process_router"] = {
                "mode": "MULTI_PALLET_RUNTIME",
                "active_order_ids": [
                    str(order["order_id"])
                    for order in orders
                    if str(order.get("status", "")) in {"RELEASED", "RUNNING"}
                ],
                "queued_orders": [
                    str(order["order_id"]) for order in orders if str(order.get("status", "")) == "QUEUED"
                ],
            }
        elif verified_queue:
            async_line["process_router"] = {
                **self.async_line_router.snapshot(),
                "mode": "VERIFIED_PHYSICAL_QUEUE",
                "active_order_id": self.v2_active_order_id,
                "queued_orders": [plan.order.order_id for _urgent, plan in self.v2_pending_plans],
            }
        else:
            async_line["process_router"] = self.async_line_router.snapshot()
        return {
            "schema_version": 2,
            "scheduler": scheduler,
            "tasks": tasks,
            "resources_v2": v2["resources_v2"],
            "zone_locks": v2["zone_locks"],
            "orders": orders,
            "faults_v2": v2["faults_v2"],
            "recoveries": v2["recoveries"],
            "manual_fault_requests": [
                *v2.get("manual_fault_requests", []),
                *(dict(hold) for hold in self._physical_fault_holds),
            ],
            "workstations": v2.get("workstations", {}),
            "async_line": async_line,
            "transfers": v2.get("transfers", {}),
            "tray_routes": v2.get("tray_routes", {}),
            "motion_plans": v2.get("motion_plans", []),
            "space_time_reservations": v2.get("space_time_reservations", []),
            "motion_blockers": v2.get("motion_blockers", {}),
            "gantt_events": v2.get("gantt_events", []),
            "furnace_v2": dict(self.v2_furnace_visual),
            "experiment_metrics": self.v2_metrics.calculate(self.manufacturing_runtime, self.scene.time),
        }

    def _v2_presentation(self, physical: dict[str, Any], active_task: Any) -> dict[str, Any]:
        wall = time.monotonic()
        if (
            self._cached_v2_snapshot is None
            or wall - self._last_v2_snapshot_wall >= self.V2_SNAPSHOT_INTERVAL_WALL_S
        ):
            self._cached_v2_snapshot = self._build_v2_presentation(physical, active_task)
            self._last_v2_snapshot_wall = wall
        return self._cached_v2_snapshot

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
        snapshot["simulation_actual_rtf"] = self.rtf_tracker.actual_rtf
        snapshot["simulation_speed_saturated"] = bool(
            self.simulation_rate.multiplier >= 2.0
            and self.rtf_tracker.actual_rtf > 0.0
            and self.rtf_tracker.actual_rtf < 0.8 * self.simulation_rate.multiplier
        )
        snapshot.update(self._v2_presentation(snapshot, active_task))
        if self.v2_pipeline_active:
            active_orders = [
                str(order.get("order_id", ""))
                for order in snapshot.get("orders", [])
                if str(order.get("status", "")) in {"RELEASED", "RUNNING"}
            ]
            queued_orders = [
                str(order.get("order_id", ""))
                for order in snapshot.get("orders", [])
                if str(order.get("status", "")) == "QUEUED"
            ]
            task_by_id = {
                str(task.get("task_id", "")): task
                for task in snapshot.get("tasks", [])
                if isinstance(task, dict)
            }
            runtime_resources = snapshot.get("resources_v2", {})
            runtime_arms: dict[str, dict[str, Any]] = {}
            for arm_id in ("ARM1", "ARM2", "ARM3"):
                resource = runtime_resources.get(arm_id, {})
                task_id = str(resource.get("current_task_id") or "")
                task = task_by_id.get(task_id, {})
                runtime_arms[arm_id.lower()] = {
                    "status": str(resource.get("status", "IDLE")).lower(),
                    "task_id": task_id,
                    "task_type": str(task.get("display_name_zh") or task.get("task_type") or ""),
                }
            snapshot["arms"] = runtime_arms
            snapshot["order_id"] = " / ".join(active_orders or queued_orders) or "-"
            snapshot["stage"] = "COMPLETE" if self.manufacturing_runtime.terminal else "PARALLEL_PRODUCTION"
            snapshot["status"] = f"动态流水：{len(active_orders)}个活动订单，{len(queued_orders)}个排队订单"
            snapshot["paused"] = self.manufacturing_runtime.paused
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
                    "comb_removal_progress": 0.0,
                    "moving": False,
                    "prefetch_unit_index": None,
                    "prefetch_complete_index": None,
                    "parallel_axes": [],
                    "parallel_active": False,
                },
            )
        self.shared.update_prepared(snapshot, viewer_running=viewer_running)

    def render_camera(self) -> None:
        assert self.coordinator is not None
        try:
            frame = self.scene.camera_rgb(
                int(self.args.camera_width),
                int(self.args.camera_height),
                "arm3_wrist_camera",
            )
            active = self.camera_inspection_active()
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
        if self.v2_pipeline_active:
            execution = self.manufacturing_runtime.executor.active.get("ARM3")
            return bool(
                execution is not None
                and execution.task.task_type.value
                in {
                    "INSPECT_BRAZING",
                    "INSPECT_FINS",
                    "POST_BRAZE_INSPECTION",
                    "SECOND_POST_BRAZE_VIEW",
                    "VERIFY_BASE_ALIGNMENT",
                }
            )
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

    def advance_simulation_frame(
        self,
        chunk_callback: Callable[[int, int], None] | None = None,
        *,
        max_chunk_steps: int | None = None,
    ) -> int:
        """Advance the real coordinator and MuJoCo model by one speed budget."""

        steps = self.simulation_rate.steps_for_frame()
        chunk_size = steps if max_chunk_steps is None else max(1, int(max_chunk_steps))
        completed = 0
        while completed < steps:
            current_chunk = min(chunk_size, steps - completed)
            for _ in range(current_chunk):
                self.tick()
            completed += current_chunk
            if chunk_callback is not None:
                chunk_callback(completed, steps)
        return steps

    def viewer_step_chunk(self) -> int:
        """Return a responsive chunk without redrawing after every few steps."""

        multiplier = float(self.simulation_rate.multiplier)
        if multiplier >= 16.0:
            return 32
        if multiplier >= 8.0:
            return 16
        if multiplier >= 4.0:
            return 8
        return 4

    def viewer_sync_fps(self) -> float:
        """Trade temporal redraw rate—not model quality—for high-speed RTF."""

        multiplier = float(self.simulation_rate.multiplier)
        if multiplier >= 32.0:
            return 10.0
        if multiplier >= 16.0:
            return 15.0
        if multiplier >= 8.0:
            return 24.0
        return self.VIEWER_SYNC_FPS

    def state_publish_interval(self) -> float:
        """Avoid serializing a large DAG faster than the UI can consume it."""

        multiplier = float(self.simulation_rate.multiplier)
        if multiplier >= 16.0:
            return 0.16
        if multiplier >= 8.0:
            return 0.10
        return 0.08

    def run_headless(self) -> int:
        assert self.coordinator is not None and self.batch_coordinator is not None
        while self.running and self.scene.time <= float(self.args.max_sim_time):
            self.drain_commands()
            self.advance_simulation_frame()
            self.service_simulation_frame()
            wall = time.monotonic()
            self.rtf_tracker.observe(self.scene.time, wall)
            if wall - self.last_publish_wall >= 0.05:
                self.publish(False)
                self.last_publish_wall = wall
            terminal = (
                self.manufacturing_runtime.terminal
                if self.v2_pipeline_active
                else self.batch_coordinator.terminal if self.batch_active() else self.coordinator.terminal
            )
            if terminal:
                break
        terminal = (
            self.manufacturing_runtime.terminal
            if self.v2_pipeline_active
            else self.batch_coordinator.terminal if self.batch_active() else self.coordinator.terminal
        )
        if not terminal and self.v2_pipeline_active:
            self.manufacturing_runtime.last_error = "headless simulation timeout"
        elif not terminal and self.current_product() is not None:
            if self.batch_active() and self.batch_coordinator.batch is not None:
                self.batch_coordinator.batch.fail("headless simulation timeout", self.scene.time)
            else:
                self.current_product().fail("headless simulation timeout", self.scene.time)
        self.publish(False)
        if self.v2_pipeline_active:
            physical = self.coordinator.snapshot(self.scene.time)
            snapshot = self._build_v2_presentation(physical, None)
            snapshot["last_error"] = self.manufacturing_runtime.last_error
        else:
            snapshot = (
                self.batch_coordinator.snapshot(self.scene.time)
                if self.batch_active()
                else self.coordinator.snapshot(self.scene.time)
            )
        print(json.dumps(jsonable(snapshot), ensure_ascii=False, indent=2), flush=True)
        if self.v2_pipeline_active:
            failed = any(
                order.get("status") in {"MANUAL_REVIEW", "CANCELLED"} for order in snapshot.get("orders", [])
            )
            return 2 if failed or bool(self.manufacturing_runtime.last_error) else 0
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
            cinematic = Path(self.args.xml).stem.endswith("_cinematic")
            viewer.cam.lookat[:] = [0.0, 0.55, 0.38] if cinematic else [-0.05, 0.62, 0.25]
            viewer.cam.distance = 3.25 if cinematic else 2.90
            viewer.cam.azimuth = -90 if cinematic else 90
            viewer.cam.elevation = -24 if cinematic else -32
            self.render_scheduler.observe_view(viewer.cam, time.monotonic())
            viewer_pacer = ViewerFramePacer(self.VIEWER_SYNC_FPS)
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

                def sync_viewer_chunk(_completed: int, _total: int) -> None:
                    # Commands, state publishing and OpenGL have independent
                    # cadences.  In particular, never turn a 32x physics
                    # budget into eight expensive viewer.sync() calls.
                    self.service_simulation_frame()
                    chunk_wall = time.monotonic()
                    self.rtf_tracker.observe(self.scene.time, chunk_wall)
                    if chunk_wall - self.last_publish_wall >= self.state_publish_interval():
                        self.publish(True)
                        self.last_publish_wall = chunk_wall
                    viewer_pacer.set_fps(self.viewer_sync_fps())
                    if viewer_pacer.due(chunk_wall):
                        viewer.sync()
                        synced_wall = time.monotonic()
                        viewer_pacer.mark_synced(synced_wall)
                        self.render_scheduler.observe_view(viewer.cam, synced_wall)

                steps = self.advance_simulation_frame(
                    sync_viewer_chunk,
                    max_chunk_steps=self.viewer_step_chunk(),
                )
                if steps == 0:
                    sync_viewer_chunk(0, 0)
                wall = time.monotonic()
                if self.render_scheduler.camera_due(
                    wall,
                    inspection_active=self.camera_inspection_active(),
                    speed_multiplier=self.simulation_rate.multiplier,
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
        elif args.orders_file:
            for plan in application.process_plans:
                application.enqueue_v2_plan(plan)
        return application.run_headless() if args.headless else application.run_viewer()
    finally:
        application.close()


if __name__ == "__main__":
    raise SystemExit(main())
