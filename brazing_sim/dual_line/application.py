"""Shared-control facade for the V2 runtime and Qt/HTTP application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from brazing_sim.api import SharedState, start_http_server
from brazing_sim.flexible import build_custom_plan

from .presentation import V2StatePresenter
from .runtime import DualLineRuntime
from .scene_adapter import DualLineSceneAdapter


@dataclass(slots=True)
class V2ControlSurface:
    """Translate common V1 API commands without owning MuJoCo or HTTP."""

    runtime: DualLineRuntime
    simulation_speed: float = 1.0
    running: bool = True
    # Order fields the last submission could not honour, surfaced to the console.
    ignored_order_fields: list[str] = field(default_factory=list)

    MINIMUM_SPEED = 0.25
    MAXIMUM_SPEED = 32.0

    def _due_at(self, command: Mapping[str, Any]) -> float | None:
        direct = command.get("due_at")
        if isinstance(direct, (int, float)):
            return float(direct)
        value = command.get("due_time")
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace("Z", "+00:00")
        try:
            due_time = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("V2 交期必须为 ISO-8601 时间") from exc
        if due_time.tzinfo is None:
            due_time = due_time.astimezone()
        now = datetime.now().astimezone()
        remaining_s = max(0.0, (due_time - now).total_seconds())
        return self.runtime.sim_time + remaining_s

    def _submit(self, command: Mapping[str, Any], *, quantity: int | None = None) -> None:
        # Fields the V1 console sends that V2 cannot honour are reported rather
        # than silently dropped: a request that appears to succeed but changed
        # nothing is worse than a clear refusal.
        ignored: list[str] = []
        if command.get("preferred_rack_layer") is not None:
            # V2 assigns furnace layers by loading position (capacity-1-index),
            # so a preferred layer cannot be reserved.
            ignored.append("首选料架层（V2 按装炉顺序分配层位）")
        strategy = str(command.get("route_strategy") or "").strip().upper()
        if strategy and strategy != "STANDARD":
            raise ValueError(f"V2 当前不支持工艺路线 {strategy}，订单未加入")
        requested_quantity = int(command.get("quantity", 1) if quantity is None else quantity)
        custom_product = command.get("custom_product")
        if custom_product is not None:
            if not isinstance(custom_product, Mapping):
                raise ValueError("V2 自定义产品参数必须是对象")
            identifier = str(command.get("order_id") or "").strip()
            if not identifier:
                identifier = self.runtime.next_order_id
            plan = build_custom_plan(
                order_id=identifier,
                quantity=requested_quantity,
                priority=int(command.get("priority", 10)),
                due_time=command.get("due_time"),
                preferred_rack_layer=command.get("preferred_rack_layer"),
                product=dict(custom_product),
                route_strategy=strategy or "STANDARD",
            )
            self.runtime.submit_plan(
                plan,
                due_at=self._due_at(command),
                urgent=bool(command.get("urgent", False)),
            )
        else:
            self.runtime.submit_order(
                str(command.get("preset", "A")),
                order_id=str(command.get("order_id") or "").strip() or None,
                quantity=requested_quantity,
                priority=int(command.get("priority", 10)),
                due_at=self._due_at(command),
                urgent=bool(command.get("urgent", False)),
            )
        if ignored:
            self.ignored_order_fields = ignored
            raise ValueError("订单已接受，但以下设置在 V2 下未生效：" + "；".join(ignored))

    def process(self, command: Mapping[str, Any]) -> None:
        kind = str(command.get("type", "")).strip()
        if kind in {"order", "order_insert"}:
            self._submit(command)
        elif kind == "batch":
            self._submit(command, quantity=int(command.get("layers", 3)))
        elif kind == "stop":
            self.runtime.pause()
        elif kind == "continue":
            self.runtime.continue_run()
        elif kind == "speed":
            action = str(command.get("action", "")).strip().lower()
            if action == "accelerate":
                self.simulation_speed = min(self.MAXIMUM_SPEED, self.simulation_speed * 2.0)
            elif action == "decelerate":
                self.simulation_speed = max(self.MINIMUM_SPEED, self.simulation_speed / 2.0)
            else:
                raise ValueError("speed action must be accelerate or decelerate")
        elif kind == "reset":
            self.runtime.reset()
            self.simulation_speed = 1.0
        elif kind == "manual_fault_inject":
            self.runtime.inject_fault(
                str(command.get("fault_type", "")),
                target=str(command.get("target", "")),
                severity=str(command.get("severity", "recoverable")),
                auto_recover=bool(command.get("auto_recover", True)),
                duration_s=command.get("duration_s"),
                label_zh=str(command.get("label_zh", "")),
            )
        elif kind == "resource_recover":
            resource = str(command.get("resource_id", ""))
            if not self.runtime.recover_resource(resource):
                raise ValueError(f"资源 {resource} 当前未被故障隔离")
        elif kind == "resource_fault":
            self.runtime.inject_fault(
                "ARM_UNAVAILABLE",
                target=str(command.get("resource_id", "")),
                duration_s=command.get("duration_s"),
                label_zh=str(command.get("fault_code", "OPERATOR_FAULT")),
            )
        elif kind == "recovery_action":
            recovery_id = str(command.get("recovery_id", ""))
            action = str(command.get("action", ""))
            if not self.runtime.recovery_action(recovery_id, action):
                raise ValueError(f"恢复计划 {recovery_id} 无法执行动作 {action}")
        elif kind == "flexibility_demo":
            self._run_flexibility_demo(str(command.get("demo", "")))
        elif kind == "segment":
            segment = str(command.get("segment", ""))
            raise RuntimeError(f"V2 单段 {segment!r} 的物理 actor 尚未接通")
        elif kind == "quit":
            self.running = False
        else:
            raise ValueError(f"unsupported V2 command: {kind or '<empty>'}")

    def _demo_order_id(self, label: str) -> str:
        base = f"FLEX_{label.upper()}"
        if base not in self.runtime.orders:
            return base
        index = 2
        while f"{base}_{index:02d}" in self.runtime.orders:
            index += 1
        return f"{base}_{index:02d}"

    def _run_flexibility_demo(self, demo: str) -> None:
        """Run only demonstrations that the V2 physical runtime can honour."""

        if demo == "product_mix":
            for preset in ("A", "B", "C"):
                self.runtime.submit_order(
                    preset,
                    order_id=self._demo_order_id(f"PRODUCT_{preset}"),
                    quantity=1,
                )
        elif demo == "resource_parallel":
            self.runtime.submit_order(
                "A",
                order_id=self._demo_order_id("DUAL_BRANCH"),
                quantity=2,
                priority=20,
            )
        elif demo == "batch_three":
            self.runtime.submit_order(
                "A",
                order_id=self._demo_order_id("BATCH_3"),
                quantity=3,
            )
        elif demo == "urgent_insert":
            self.runtime.submit_order("B", order_id=self._demo_order_id("NORMAL_B"), quantity=2)
            self.runtime.submit_order(
                "C",
                order_id=self._demo_order_id("URGENT_C"),
                quantity=1,
                priority=99,
                urgent=True,
            )
        elif demo == "fault_loop":
            self.runtime.submit_order(
                "A",
                order_id=self._demo_order_id("FAULT_LOOP"),
                quantity=1,
            )
            self.runtime.inject_fault(
                "BRAZING_MISSING",
                target="path_02",
                label_zh="柔性演示：漏涂闭环",
            )
        else:
            raise ValueError(f"unsupported V2 flexibility demo: {demo or '-'}")


class V2BrazingApplication:
    """Own V2 simulation state while reusing the V1 HTTP/Qt control shell."""

    VIEWER_FPS = 30.0

    def __init__(self, args: Any) -> None:
        self.args = args
        self.scene = DualLineSceneAdapter(Path(args.xml))
        if bool(args.headless) and bool(args.fast):
            # Fast headless runs validate logical/physical completion rather
            # than contact-rich viewer cinematics.  A 10 ms MuJoCo step keeps
            # the position-controlled V1-compatible paths deterministic while
            # avoiding 25 substeps for every 50 ms application tick.  Viewer
            # sessions retain the authored 2 ms physics step.
            self.scene.model.opt.timestep = min(0.010, float(args.dt))
        self.runtime = DualLineRuntime(fast=bool(args.fast))
        self.controls = V2ControlSurface(self.runtime)
        self.presenter = V2StatePresenter()
        self.shared = SharedState()
        self.server: Any | None = None
        self.ui_process: subprocess.Popen[Any] | None = None
        self._step_credit = 0.0
        self._last_wall = time.monotonic()
        self._last_sim_time = 0.0
        self.actual_rtf = 0.0
        self.last_error = ""
        self._last_camera_render_wall = float("-inf")
        self.scene.sync(self.runtime)
        self.publish(viewer_running=False)

    def submit_cli_orders(self) -> None:
        for index, preset in enumerate(self.args.order_presets, start=1):
            self.runtime.submit_order(
                preset,
                order_id=f"V2_{preset}_{index:02d}",
                priority=10 + index,
            )
        self.scene.sync(self.runtime)
        self.publish(viewer_running=False)

    def start_services(self) -> None:
        self.server = start_http_server(self.shared, self.args.host, int(self.args.port))
        actual_port = int(self.server.server_address[1])
        print(f"[V2 HTTP] http://{self.args.host}:{actual_port}", flush=True)
        if not self.args.no_ui:
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[2] / "brazing_line_v2.py"),
                "--ui-client",
                "--host",
                str(self.args.host),
                "--port",
                str(actual_port),
            ]
            self.ui_process = subprocess.Popen(command)

    def process_command(self, command: Mapping[str, Any]) -> None:
        try:
            self.controls.process(command)
            self.last_error = ""
            self.scene.sync(self.runtime)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.last_error = str(exc)

    def drain_commands(self) -> None:
        while True:
            try:
                command = self.shared.commands.get_nowait()
            except queue.Empty:
                return
            self.process_command(command)

    def _update_rtf(self) -> None:
        wall = time.monotonic()
        wall_delta = wall - self._last_wall
        sim_time = float(self.runtime.sim_time)
        if wall_delta > 1.0e-9:
            measured = max(0.0, sim_time - self._last_sim_time) / wall_delta
            self.actual_rtf = measured if self.actual_rtf <= 0.0 else 0.8 * self.actual_rtf + 0.2 * measured
        self._last_wall = wall
        self._last_sim_time = sim_time

    def advance_frame(self) -> int:
        if self.runtime.paused:
            return 0
        self._step_credit += self.controls.simulation_speed
        steps = int(self._step_credit + 1.0e-12)
        self._step_credit -= steps
        for _ in range(steps):
            if not self.runtime.complete:
                self.runtime.tick(float(self.args.dt))
            self.scene.sync(self.runtime)
            self.scene.step_physics(float(self.args.dt))
        self._update_rtf()
        return steps

    def publish(self, *, viewer_running: bool) -> dict[str, Any]:
        state = self.presenter.present(
            self.runtime.snapshot(),
            simulation_speed=self.controls.simulation_speed,
            actual_rtf=self.actual_rtf,
            viewer_running=viewer_running,
        )
        state["last_error"] = self.last_error
        robot_motion = self.scene.robot_motion_snapshot()
        state["robot_motion"] = robot_motion
        state["inspections"] = self.scene.inspection_snapshot()
        for arm_name, physical in robot_motion.items():
            arm = state.get("arms", {}).get(arm_name)
            if not isinstance(arm, dict):
                continue
            arm.update(
                {
                    "physical_mode": physical.get("mode"),
                    "planner": physical.get("planner"),
                    "target_zh": physical.get("target_zh"),
                    "waypoint_index": physical.get("waypoint_index", 0),
                    "waypoint_count": physical.get("waypoint_count", 0),
                    "position_error_m": physical.get("position_error_m", 0.0),
                    "orientation_error_rad": physical.get(
                        "orientation_error_rad",
                        0.0,
                    ),
                    "joint_positions": physical.get("joint_positions", []),
                    "physical_complete": physical.get("physical_complete", False),
                    "error": physical.get("failure", ""),
                }
            )
        # Replace duration-only task progress with measured physical waypoint
        # progress whenever a robot actor is active.  Logical duration may hit
        # zero while IK motion or camera analysis is still settling; reporting
        # 100% at that point was the main reason the task graph looked stale.
        for task in state.get("tasks", []):
            if not isinstance(task, dict) or task.get("status") != "RUNNING":
                continue
            resource = str(task.get("assigned_resource") or "").lower()
            physical = robot_motion.get(resource)
            if not isinstance(physical, dict):
                continue
            waypoint_count = int(physical.get("waypoint_count", 0))
            if waypoint_count <= 0:
                continue
            # The projector measures both the completed waypoint count and the
            # active trajectory segment.  Using only ``waypoint_index`` made the
            # task graph jump in coarse steps and look stale between waypoints.
            progress = min(1.0, max(0.0, float(physical.get("progress", 0.0))))
            task["progress"] = progress
            task["updated_at"] = float(self.runtime.sim_time)
            detail = str(task.get("display_detail_zh", "")).split(" · 执行进度", 1)[0]
            target = str(physical.get("target_zh") or "")
            task["display_detail_zh"] = (
                f"{detail} · {target} · 执行进度 {progress * 100.0:.0f}%"
                if target
                else f"{detail} · 执行进度 {progress * 100.0:.0f}%"
            )
        state["motion_plans"] = [
            {
                "resource_id": arm_name.upper(),
                "request_id": physical.get("operation", ""),
                "planner": physical.get("planner", ""),
                "start_time": 0.0,
                "end_time": 0.0,
                "waiting_time": 0.0,
                "reservation_id": (physical.get("operation") if physical.get("operation") else None),
            }
            for arm_name, physical in robot_motion.items()
            if physical.get("operation")
        ]
        state["async_line"]["minimum_tray_clearance_m"] = self.scene.visible_tray_clearance_m()
        physical_owners = self.scene.physical_owner_snapshot()
        state["async_line"]["physical_tray_owners"] = physical_owners
        for tray_id, route in state.get("tray_routes", {}).items():
            if isinstance(route, dict):
                route["physical_owner"] = physical_owners.get(str(tray_id))
        transfers = self.scene.transport_snapshot()
        state["transfers"] = transfers
        state["async_line"]["transfer_positions_m"] = {
            transfer_id: float(item["progress"]) * float(item["distance_m"])
            for transfer_id, item in transfers.items()
        }
        state["transport_mechanisms"] = self.scene.route_mechanism_snapshot()
        furnace_transfer = self.scene.furnace_transfer_snapshot()
        if transfers:
            active_transfer = next(iter(transfers.values()))
            state["conveyor"] = {
                "phase": active_transfer["route_id"],
                "position_m": float(active_transfer["progress"]) * float(active_transfer["distance_m"]),
                "travel_m": float(active_transfer["distance_m"]),
                "progress": float(active_transfer["progress"]),
                "moving": True,
                "segment_index": active_transfer.get("segment_index", 1),
                "segment_count": active_transfer.get("segment_count", 1),
            }
            state["transfer"] = {
                "phase": active_transfer["status"],
                "step": active_transfer["route_id"],
                "unit_id": active_transfer["tray_id"],
                "conveyor_progress": active_transfer["progress"],
                "lift_height_m": max(
                    furnace_transfer["lift"],
                    furnace_transfer["rear_lift"],
                ),
                "pusher_position_m": furnace_transfer["pusher"],
                "rear_extractor_position_m": furnace_transfer["rear_extractor"],
                "lock_position_m": max(
                    furnace_transfer["pusher"],
                    furnace_transfer["rear_extractor"],
                ),
                "moving": True,
            }
        else:
            state["transfer"].update(
                {
                    "lift_height_m": max(
                        furnace_transfer["lift"],
                        furnace_transfer["rear_lift"],
                    ),
                    "pusher_position_m": furnace_transfer["pusher"],
                    "rear_extractor_position_m": furnace_transfer["rear_extractor"],
                }
            )
        state["scene"] = {
            "compiled": True,
            "xml": str(self.scene.xml_path),
            "mujoco_time": float(self.scene.data.time),
            "nbody": int(self.scene.model.nbody),
            "nu": int(self.scene.model.nu),
        }
        self.shared.update_prepared(state)
        return state

    def update_dual_camera(self) -> None:
        now = time.monotonic()
        operations = self.runtime.operations
        inspection_active = any(
            operation.kind
            in {
                "MATERIAL_INSPECTION",
                "PRE_BRAZE_INSPECTION",
                "POST_BRAZE_INSPECTION",
            }
            for operation in operations.values()
        )
        target_fps = 8.0 if inspection_active else 2.0
        if self.controls.simulation_speed >= 16.0:
            target_fps = min(target_fps, 1.0)
        elif self.controls.simulation_speed >= 8.0:
            target_fps = min(target_fps, 2.0)
        if now - self._last_camera_render_wall < 1.0 / target_fps:
            return
        try:
            frame = self.scene.dual_camera_rgb(width=480, height=360)
        except Exception as exc:  # OpenGL availability is platform-owned.
            self.shared.update(
                camera_active=False,
                camera_status=f"V2 双相机等待图形上下文：{exc}",
            )
            self._last_camera_render_wall = now
            return
        height, width, channels = frame.shape
        if channels != 3:
            raise RuntimeError("V2 dual camera renderer must return RGB")
        ppm = (
            f"P6\n{width} {height}\n255\n".encode("ascii")
            + frame.astype(
                "uint8",
                copy=False,
            ).tobytes()
        )
        inspections = self.scene.inspection_snapshot()
        active_analysis = next(
            (item for item in reversed(inspections) if not bool(item.get("analysis_complete", False))),
            None,
        )
        if active_analysis is None:
            camera_status = "左：Arm3 混合末端相机　|　右：焊后固定相机"
        else:
            camera_status = (
                "已对齐整体轮廓并拍照，后台分析 "
                f"{float(active_analysis['analysis_elapsed_s']):.1f}/"
                f"{float(active_analysis['analysis_seconds']):.1f} s"
            )
        self.shared.update_camera(
            ppm,
            width=width,
            height=height,
            active=inspection_active,
            status=camera_status,
        )
        self._last_camera_render_wall = now

    def run_headless(self) -> int:
        if not self.args.order_presets:
            print(json.dumps(self.publish(viewer_running=False), ensure_ascii=False, indent=2))
            return 0
        while (
            self.controls.running
            and (not self.runtime.complete or not self.scene.transport_settled)
            and max(self.runtime.sim_time, float(self.scene.data.time)) + 1.0e-12
            < float(self.args.max_sim_time)
        ):
            self.advance_frame()
        state = self.publish(viewer_running=False)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if self.runtime.complete and self.scene.transport_settled else 2

    def run_viewer(self) -> int:
        import mujoco.viewer

        model, data = self.scene.model, self.scene.data
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=bool(self.args.show_mujoco_ui),
            show_right_ui=bool(self.args.show_mujoco_ui),
        ) as viewer:
            viewer.cam.lookat[:] = [1.75, 0.0, 0.30]
            viewer.cam.distance = 5.6
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -25
            self.shared.update(viewer_running=True)
            frame_period = 1.0 / self.VIEWER_FPS
            while viewer.is_running() and self.controls.running:
                started = time.monotonic()
                self.drain_commands()
                self.advance_frame()
                self.publish(viewer_running=True)
                viewer.sync()
                self.update_dual_camera()
                delay = frame_period - (time.monotonic() - started)
                if delay > 0.0:
                    time.sleep(delay)
        self.publish(viewer_running=False)
        return 0

    def close(self) -> None:
        self.shared.update(status="shutdown", viewer_running=False, camera_active=False)
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.ui_process is not None and self.ui_process.poll() is None:
            self.ui_process.terminate()
        self.scene.close()


__all__ = ["V2BrazingApplication", "V2ControlSurface"]
