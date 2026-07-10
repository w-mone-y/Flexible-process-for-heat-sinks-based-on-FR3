"""Three-FR3 flexible production line for low-voltage electrical boards.

Layout (project plan section 3.2):
  left   Table1 feed rack (scattered parts) -> Arm1 (camera-guided feeding)
  center shared space                        -> Arm2 (core process arm:
                                                quick-change gripper-press +
                                                electric screwdriver)
  right  inspection/rework                   -> Arm3 (inspection, rework)

Per-component process chain (plan sections 5.1-5.4):
  feed_pick -> feed_place(type fixture) -> assemble_pick -> assemble_place
  -> press -> screw x N -> inspect [-> rework(press / screw) -> re-inspect]

Modules:
  object_manager.py  component specs/orders/state machine + simulated camera
  skill_library.py   pose math + skill-step generators (incl. tool offsets)
  arm_controller.py  single-arm control law + skill executor + parking
  arm2_tool_manager.py  Arm2 quick-change tool welds + get/return skills
  (this file)        shared state, HTTP/UI/terminal, scheduler, main loop

Run:  mjpython multi_arm_line.py [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import numpy as np

from arm2_tool_manager import Arm2ToolManager
from arm_controller import ArmController
from object_manager import (
    COMPONENT_INSTANCES,
    ORDER_PRESETS,
    TYPE_STAGING_SITE,
    ObjectManager,
)
from skill_library import (
    GRIPPER_OPEN_CTRL,
    INSPECT_XY_TOL,
    INSPECT_YAW_TOL_DEG,
    SkillStep,
    gripper_close_ctrl_for_width,
    inspect_steps,
    nearest_symmetric_yaw,
    pick_steps,
    place_steps,
    pose_array_to_command,
    press_steps,
    screw_steps,
    yaw_error_with_symmetry,
    yaw_from_mat,
)

ARM_NAMES = ("arm1", "arm2", "arm3")
ZONE_CENTRAL = "central"
ZONE_STAGING = "staging"
BOARD_TOP_Z = 0.110
MAX_REWORK_ROUNDS = 2


# ---------------------------------------------------------------------------
# Pipeline data model.
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    name: str                  # feed_pick/.../get_tool/return_tool/press/screw/inspect
    arm: str
    tokens: tuple[str, ...] = ()
    screw: str = ""
    tool: str = ""             # gripper_press | screwdriver (tool-change stages)
    status: str = "pending"    # pending -> active -> done


@dataclass
class Job:
    job_id: str
    order_id: str
    component: str
    slot: str
    screws: list[str]
    inspect: bool
    stages: list[Stage] = field(default_factory=list)
    stage_idx: int = 0
    staging_site: str = ""
    attach_feed: dict[str, np.ndarray] = field(default_factory=dict)
    attach_asm: dict[str, np.ndarray] = field(default_factory=dict)
    inspect_result: str = ""
    rework_count: int = 0

    def current_stage(self) -> Optional[Stage]:
        if 0 <= self.stage_idx < len(self.stages):
            return self.stages[self.stage_idx]
        return None

    @property
    def done(self) -> bool:
        return self.stage_idx >= len(self.stages)


def build_job_stages(job: Job) -> list[Stage]:
    """Arm2 quick-change flow (design doc sections 5, 8):
    get gripper_press -> pick/place/press/release/fixture -> return gripper_press
    -> get screwdriver -> tighten screws -> return screwdriver -> inspect."""
    stages = [
        Stage("feed_pick", "arm1"),
        Stage("feed_place", "arm1", tokens=(ZONE_STAGING,)),
        Stage("get_tool", "arm2", tool="gripper_press"),
        Stage("assemble_pick", "arm2", tokens=(ZONE_STAGING,)),
        Stage("assemble_place", "arm2", tokens=(ZONE_CENTRAL,)),
        Stage("press", "arm2", tokens=(ZONE_CENTRAL,)),
        Stage("return_tool", "arm2", tool="gripper_press"),
        Stage("get_tool", "arm2", tool="screwdriver"),
    ]
    for screw in job.screws:
        stages.append(Stage("screw", "arm2", tokens=(ZONE_CENTRAL,), screw=screw))
    stages.append(Stage("return_tool", "arm2", tool="screwdriver"))
    if job.inspect:
        stages.append(Stage("inspect", "arm3", tokens=(ZONE_CENTRAL,)))
    return stages


# ---------------------------------------------------------------------------
# Shared state + HTTP + terminal + UI.
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self.status = "starting"
        self.viewer_running = False
        self.show_ee_axes = True
        self.show_mocap = True
        self.arms: dict[str, dict[str, Any]] = {
            name: {"stage": "", "job": "", "step": "", "held": "", "pos_err": 0.0} for name in ARM_NAMES
        }
        self.zone_owners: dict[str, str] = {}
        self.order_id = ""
        self.order_elapsed = 0.0
        self.jobs: list[dict[str, Any]] = []
        self.screw_states: dict[str, str] = {}
        self.component_states: dict[str, str] = {}
        self.inspect_results: list[dict[str, Any]] = []
        self.fault_screws: list[str] = []
        self.available_orders = sorted(ORDER_PRESETS.keys())
        self.server_time = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": str(self.status),
                "viewer_running": bool(self.viewer_running),
                "show_ee_axes": bool(self.show_ee_axes),
                "show_mocap": bool(self.show_mocap),
                "arms": json.loads(json.dumps(self.arms)),
                "zone_owners": dict(self.zone_owners),
                "order_id": str(self.order_id),
                "order_elapsed": float(self.order_elapsed),
                "jobs": json.loads(json.dumps(self.jobs)),
                "screw_states": dict(self.screw_states),
                "component_states": dict(self.component_states),
                "inspect_results": json.loads(json.dumps(self.inspect_results)),
                "fault_screws": list(self.fault_screws),
                "available_orders": list(self.available_orders),
                "server_time": float(self.server_time),
            }

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.server_time = time.time()


class RequestHandler(BaseHTTPRequestHandler):
    shared: SharedState

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/state":
            self._send_json(self.shared.snapshot())
        else:
            self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/order":
                preset = str(payload.get("preset", ""))
                if preset:
                    order = ORDER_PRESETS.get(preset.upper())
                    if order is None:
                        raise ValueError(f"unknown order preset: {preset}")
                else:
                    order = payload
                if not order.get("components"):
                    raise ValueError("order requires components")
                self.shared.commands.put({"type": "order", "order": order})
                self._send_json({"ok": True, "order_id": order.get("order_id", "custom")})
            elif self.path == "/fault":
                screw = str(payload.get("screw", ""))
                component = str(payload.get("component", ""))
                if not screw and not component:
                    raise ValueError("fault requires screw or component, e.g. {'screw': 'screw_slot_1_a'} or {'component': 'relay_1'}")
                self.shared.commands.put({"type": "fault", "screw": screw, "component": component})
                self._send_json({"ok": True, "screw": screw, "component": component})
            elif self.path == "/scatter":
                seed = payload.get("seed", None)
                self.shared.commands.put({"type": "scatter", "seed": seed})
                self._send_json({"ok": True, "seed": seed})
            elif self.path == "/move":
                arm = str(payload.get("arm", ""))
                waypoints = payload.get("waypoints", [])
                if arm not in ARM_NAMES:
                    raise ValueError(f"arm must be one of {ARM_NAMES}")
                if not isinstance(waypoints, list) or len(waypoints) != 1:
                    raise ValueError("move requires exactly one waypoint")
                self.shared.commands.put({"type": "move", "arm": arm, "waypoints": waypoints})
                self._send_json({"ok": True})
            elif self.path == "/stop":
                self.shared.commands.put({"type": "stop"})
                self._send_json({"ok": True})
            elif self.path == "/ee_axes":
                visible = bool(payload.get("visible", True))
                self.shared.update(show_ee_axes=visible)
                self._send_json({"ok": True, "visible": visible})
            elif self.path == "/mocap":
                visible = bool(payload.get("visible", True))
                self.shared.update(show_mocap=visible)
                self._send_json({"ok": True, "visible": visible})
            elif self.path == "/quit":
                self.shared.commands.put({"type": "quit"})
                self._send_json({"ok": True})
            else:
                self._send_json({"ok": False, "error": "not found"}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)


def start_http_server(shared: SharedState, host: str, port: int) -> ThreadingHTTPServer:
    handler_cls = type("MultiArmRequestHandler", (RequestHandler,), {"shared": shared})
    server = ThreadingHTTPServer((host, port), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def find_python_for_ui(requested: Optional[str]) -> str:
    if requested:
        return requested
    candidates: list[str] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "bin" / "python"))
    if shutil.which("python3"):
        candidates.append(str(shutil.which("python3")))
    candidates.append(sys.executable)
    for candidate in candidates:
        if candidate and Path(candidate).exists() and "mjpython" not in Path(candidate).name.lower():
            return candidate
    return sys.executable


def post_json(url: str, payload: dict[str, Any], timeout: float = 1.0) -> dict[str, Any]:
    from urllib import request

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: float = 1.0) -> dict[str, Any]:
    from urllib import request

    with request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def run_ui_client(args: argparse.Namespace) -> int:
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print("没有找到 PySide6。请先安装 PySide6。", file=sys.stderr, flush=True)
        return 2

    base_url = f"http://127.0.0.1:{int(args.port)}"

    class ControlPanel(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("multi_arm_line - 三臂 FR3 柔性产线")
            layout = QVBoxLayout(self)

            order_group = QGroupBox("订单 / Orders")
            order_layout = QHBoxLayout(order_group)
            for preset, label in (("A", "订单 A 基础型 (2件2丝1检)"), ("B", "订单 B 复杂型 (3件4丝2检)"), ("C", "订单 C 急单 (1件2丝1检)")):
                btn = QPushButton(label)
                btn.clicked.connect(lambda _=False, p=preset: self.post("/order", {"preset": p}))
                order_layout.addWidget(btn)
            scatter_btn = QPushButton("重新散乱料架")
            scatter_btn.clicked.connect(lambda: self.post("/scatter", {}))
            order_layout.addWidget(scatter_btn)
            stop_btn = QPushButton("停止/清空")
            stop_btn.clicked.connect(lambda: self.post("/stop", {}))
            order_layout.addWidget(stop_btn)
            order_layout.addStretch(1)
            layout.addWidget(order_group)

            fault_group = QGroupBox("故障注入（螺丝: screw_slot_1_a -> 重拧返工；元件: relay_1 -> 偏位重压返工）")
            fault_layout = QHBoxLayout(fault_group)
            self.fault_input = QLineEdit()
            self.fault_input.setPlaceholderText("screw_slot_1_a 或 relay_1")
            fault_layout.addWidget(self.fault_input, 1)
            fault_btn = QPushButton("注入故障")
            fault_btn.clicked.connect(self.inject_fault)
            fault_layout.addWidget(fault_btn)
            layout.addWidget(fault_group)

            arms_group = QGroupBox("三臂状态 / Arm Status")
            arms_layout = QVBoxLayout(arms_group)
            self.arm_labels: dict[str, QLabel] = {}
            for name, duty in (("arm1", "相机感知+散乱取放"), ("arm2", "核心工艺: 装配+压装+电批紧固"), ("arm3", "检测返工")):
                label = QLabel(f"{name} ({duty}): waiting...")
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                self.arm_labels[name] = label
                arms_layout.addWidget(label)
            self.zone_label = QLabel("zones: -")
            self.zone_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            arms_layout.addWidget(self.zone_label)
            layout.addWidget(arms_group)

            progress_group = QGroupBox("订单进度 / Progress")
            progress_layout = QVBoxLayout(progress_group)
            self.order_label = QLabel("order: -")
            self.jobs_label = QLabel("jobs: -")
            self.comp_label = QLabel("components: -")
            self.screws_label = QLabel("screws: -")
            self.inspect_label = QLabel("inspect: -")
            self.status_label = QLabel("starting...")
            for label in (self.order_label, self.jobs_label, self.comp_label, self.screws_label, self.inspect_label, self.status_label):
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                label.setWordWrap(True)
                progress_layout.addWidget(label)
            layout.addWidget(progress_group)

            vis_group = QGroupBox("Visibility")
            vis_layout = QHBoxLayout(vis_group)
            self.ee_checkbox = QCheckBox("显示末端坐标轴")
            self.ee_checkbox.setChecked(True)
            self.ee_checkbox.toggled.connect(lambda value: self.post("/ee_axes", {"visible": bool(value)}))
            self.mocap_checkbox = QCheckBox("显示 mocap targets")
            self.mocap_checkbox.setChecked(True)
            self.mocap_checkbox.toggled.connect(lambda value: self.post("/mocap", {"visible": bool(value)}))
            vis_layout.addWidget(self.ee_checkbox)
            vis_layout.addWidget(self.mocap_checkbox)
            vis_layout.addStretch(1)
            layout.addWidget(vis_group)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh_state)
            self.timer.start(120)

        def post(self, path: str, payload: dict[str, Any]) -> None:
            try:
                post_json(base_url + path, payload, timeout=0.5)
            except Exception as exc:
                self.status_label.setText(f"请求失败: {exc}")

        def inject_fault(self) -> None:
            text = self.fault_input.text().strip()
            if not text:
                return
            if text.startswith("screw_"):
                self.post("/fault", {"screw": text})
            else:
                self.post("/fault", {"component": text})

        def refresh_state(self) -> None:
            try:
                state = get_json(base_url + "/state", timeout=0.5)
                arms = state.get("arms", {})
                for name, label in self.arm_labels.items():
                    info = arms.get(name, {})
                    stage = info.get("stage") or "idle"
                    job = info.get("job") or "-"
                    step = info.get("step") or "-"
                    held = info.get("held") or "-"
                    label.setText(f"{name}: [{stage}] job={job} step={step} held={held}")
                zones = state.get("zone_owners", {})
                self.zone_label.setText(
                    "zones: " + ", ".join(f"{z}={o or 'free'}" for z, o in sorted(zones.items())) if zones else "zones: all free"
                )
                order_id = state.get("order_id") or "-"
                elapsed = float(state.get("order_elapsed", 0.0))
                self.order_label.setText(f"order: {order_id}   elapsed: {elapsed:.1f}s")
                jobs = state.get("jobs", [])
                self.jobs_label.setText(
                    "jobs: " + ("  |  ".join(f"{j['component']}->{j['slot']} [{j['progress']}]" for j in jobs) if jobs else "-")
                )
                comps = state.get("component_states", {})
                if comps:
                    self.comp_label.setText("components: " + "  ".join(f"{k}:{v}" for k, v in sorted(comps.items())))
                screws = state.get("screw_states", {})
                if screws:
                    self.screws_label.setText("screws: " + "  ".join(f"{k.replace('screw_', '')}:{v}" for k, v in sorted(screws.items())))
                else:
                    self.screws_label.setText("screws: -")
                inspections = state.get("inspect_results", [])
                if inspections:
                    self.inspect_label.setText(
                        "inspect: " + "  ".join(f"{r['slot']}:{r['result']}({r.get('reason', '')})" for r in inspections[-6:])
                    )
                else:
                    self.inspect_label.setText("inspect: -")
                self.status_label.setText(str(state.get("status", "")))
            except Exception as exc:
                self.status_label.setText(f"连接控制器失败: {exc}")

        def closeEvent(self, event) -> None:  # type: ignore[override]
            try:
                post_json(base_url + "/quit", {}, timeout=0.3)
            except Exception:
                pass
            event.accept()

    app = QApplication(sys.argv[:1])
    panel = ControlPanel()
    panel.resize(1180, 640)
    panel.show()
    print("[UI] multi_arm_line PySide6 panel opened.", flush=True)
    return int(app.exec())


def print_terminal_help() -> None:
    print(
        "\n[终端命令]\n"
        "  order_a/b/c          执行预设订单 A / B / C\n"
        "  fault <名字>         注入故障: screw_slot_1_a(重拧返工) 或 relay_1(偏位重压返工)\n"
        "  scatter [seed]       重新散乱料架元件\n"
        "  stop                 停止并清空所有任务\n"
        "  ee_axes on/off       显示/隐藏末端坐标轴\n"
        "  mocap on/off         显示/隐藏 mocap targets\n"
        "  help                 显示命令\n",
        flush=True,
    )


def start_terminal_thread(shared: SharedState) -> None:
    def loop() -> None:
        print_terminal_help()
        while True:
            line = sys.stdin.readline()
            if line == "":
                time.sleep(0.1)
                continue
            cmd = line.strip().lower().replace("-", "_")
            if not cmd:
                continue
            if cmd.startswith("order_"):
                preset = cmd.split("_", 1)[1].upper()
                order = ORDER_PRESETS.get(preset)
                if order is None:
                    print(f"[terminal] 未知订单预设: {preset}", flush=True)
                    continue
                shared.commands.put({"type": "order", "order": order})
            elif cmd.startswith("fault "):
                target = cmd.split(maxsplit=1)[1]
                if target.startswith("screw_"):
                    shared.commands.put({"type": "fault", "screw": target, "component": ""})
                else:
                    shared.commands.put({"type": "fault", "screw": "", "component": target})
            elif cmd.startswith("scatter"):
                parts = cmd.split()
                seed = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                shared.commands.put({"type": "scatter", "seed": seed})
            elif cmd == "stop":
                shared.commands.put({"type": "stop"})
            elif cmd in {"ee_axes on", "ee on"}:
                shared.update(show_ee_axes=True)
            elif cmd in {"ee_axes off", "ee off"}:
                shared.update(show_ee_axes=False)
            elif cmd in {"mocap on", "target on"}:
                shared.update(show_mocap=True)
            elif cmd in {"mocap off", "target off"}:
                shared.update(show_mocap=False)
            elif cmd == "help":
                print_terminal_help()

    threading.Thread(target=loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Pipeline world: stage builders + fixed-rule scheduler + zone mutexes.
# ---------------------------------------------------------------------------
class PipelineWorld:
    def __init__(
        self,
        model: Any,
        data: Any,
        arms: dict[str, ArmController],
        objects: ObjectManager,
        shared: SharedState,
        args: argparse.Namespace,
    ) -> None:
        self.model = model
        self.data = data
        self.arms = arms
        self.objects = objects
        self.shared = shared
        self.args = args

        self.slot_site_ids: dict[str, int] = {}
        for i in range(model.nsite):
            name = model.site(i).name
            if name and name.startswith("slot_") and "screw" not in name:
                self.slot_site_ids[name] = int(i)
        self.staging_site_ids = {name: int(model.site(name).id) for name in TYPE_STAGING_SITE.values()}

        self.tool_mgr = Arm2ToolManager(model, data, arms["arm2"])

        self.jobs: list[Job] = []
        self.zone_owner: dict[str, str] = {ZONE_CENTRAL: "", ZONE_STAGING: ""}
        self.staging_owner: dict[str, str] = {name: "" for name in TYPE_STAGING_SITE.values()}
        self.arm_job: dict[str, Optional[Job]] = {name: None for name in ARM_NAMES}
        self.fault_screws: set[str] = set()
        self.order_id = ""
        self.order_started = 0.0
        self.order_done = False
        self.status = "ready"

    def site_pose(self, site_id: int) -> tuple[np.ndarray, float]:
        pos = np.array(self.data.site(site_id).xpos, copy=True)
        yaw = yaw_from_mat(np.asarray(self.data.site(site_id).xmat, dtype=float))
        return pos, yaw

    # -- order intake ---------------------------------------------------------
    def load_order(self, order: dict[str, Any]) -> None:
        if any(not job.done for job in self.jobs):
            raise ValueError("an order is still running; stop first")
        components = order.get("components", [])
        screws = [str(s) for s in order.get("screws", [])]
        inspection_points = {str(p) for p in order.get("inspection_points", [])}

        reserved: set[str] = set()
        jobs: list[Job] = []
        for idx, entry in enumerate(components):
            type_name = str(entry.get("type", ""))
            slot = str(entry.get("target_slot", ""))
            instance = str(entry.get("instance", "") or "")
            if slot not in self.slot_site_ids:
                raise ValueError(f"unknown slot: {slot}")
            if instance:
                if instance not in COMPONENT_INSTANCES:
                    raise ValueError(f"unknown component instance: {instance}")
                if type_name and COMPONENT_INSTANCES.get(instance) != type_name:
                    raise ValueError(f"instance {instance} is not of type {type_name}")
            else:
                candidates = sorted(
                    name for name, tname in COMPONENT_INSTANCES.items() if tname == type_name and name not in reserved
                )
                if not candidates:
                    raise ValueError(f"no free instance of type {type_name}")
                instance = candidates[0]
            reserved.add(instance)
            job_screws = [s for s in screws if s.startswith(f"screw_{slot}_")]
            for s in job_screws:
                if s not in self.objects.screw_site_ids:
                    raise ValueError(f"unknown screw: {s}")
            job = Job(
                job_id=f"j{idx + 1}",
                order_id=str(order.get("order_id", "custom")),
                component=instance,
                slot=slot,
                screws=job_screws,
                inspect=(f"inspect_{slot}" in inspection_points),
            )
            job.stages = build_job_stages(job)
            jobs.append(job)

        matched = {s for job in jobs for s in job.screws}
        unmatched = [s for s in screws if s not in matched]
        if unmatched:
            raise ValueError(f"screws not matching any ordered slot: {unmatched}")

        self.jobs = jobs
        self.order_id = str(order.get("order_id", "custom"))
        self.order_started = time.time()
        self.order_done = False
        for job in jobs:
            self.objects.set_status(job.component, "raw")
            for screw in job.screws:
                self.objects.set_screw_state(screw, "loose")
        self.status = f"order {self.order_id}: {len(jobs)} jobs queued"

    def stop_all(self) -> None:
        for name, arm in self.arms.items():
            job = self.arm_job[name]
            if job is not None:
                stage = job.current_stage()
                if stage is not None and stage.status == "active":
                    stage.status = "pending"
            if arm.held_component:
                arm.detach_component(arm.held_component)
            arm.gripper_ctrl = GRIPPER_OPEN_CTRL
            arm.manual_hold = False
            arm.cancel()
            self.arm_job[name] = None
        self.jobs = []
        self.zone_owner = {ZONE_CENTRAL: "", ZONE_STAGING: ""}
        self.staging_owner = {name: "" for name in TYPE_STAGING_SITE.values()}
        self.tool_mgr.reset_to_rack()
        self.objects.fixture_release_all()
        self.objects.reset_buffer_slots()
        self.order_id = ""
        self.order_done = False
        self.status = "stopped; all tasks cleared"

    def inject_fault(self, screw: str = "", component: str = "") -> None:
        if screw:
            self.fault_screws.add(screw)
            self.status = f"fault injected: {screw} will stay loose after fastening"
        elif component:
            if component not in COMPONENT_INSTANCES:
                raise ValueError(f"unknown component: {component}")
            self.objects.perturb_component(component)
            self.status = f"fault injected: {component} displaced (next inspection will fail -> re-press)"

    # -- stage step builders -----------------------------------------------------
    def build_stage_steps(self, job: Job, stage: Stage) -> list[SkillStep]:
        spec = self.objects.spec_of(job.component)
        if stage.name == "get_tool":
            return self.tool_mgr.change_tool_steps(stage.tool)
        if stage.name == "return_tool":
            if self.tool_mgr.state["current_tool"] != stage.tool:
                return []
            return self.tool_mgr.return_tool_steps()
        if stage.name == "feed_pick":
            # Simulated camera perception of the scattered part on Table1.
            pos, yaw = self.objects.perceive(job.component)
            steps, record = pick_steps(job.component, spec, pos, yaw)
            job.attach_feed = record
            return steps
        if stage.name == "feed_place":
            site_name = TYPE_STAGING_SITE[spec.type_name]
            job.staging_site = site_name
            self.staging_owner[site_name] = job.job_id
            site_id = self.staging_site_ids[site_name]
            return place_steps(
                job.component, spec, lambda: self.site_pose(site_id), job.attach_feed, loose_tol=True, release_dwell_s=0.4
            )
        if stage.name == "assemble_pick":
            if self.tool_mgr.state["current_tool"] != "gripper_press":
                raise RuntimeError("assemble_pick requires gripper_press tool")
            site_name = TYPE_STAGING_SITE[spec.type_name]
            print(f"[Arm2] Pick {job.component} from {site_name}", flush=True)
            pos, yaw = self.objects.perceive(job.component)
            steps, record = pick_steps(job.component, spec, pos, yaw)
            job.attach_asm = record
            return steps
        if stage.name == "assemble_place":
            if self.tool_mgr.state["current_tool"] != "gripper_press":
                raise RuntimeError("assemble_place requires gripper_press tool")
            print(f"[Arm2] Move {job.component} to {job.slot}", flush=True)
            site_id = self.slot_site_ids[job.slot]
            # Must fully open fingers + detach weld BEFORE retreat, otherwise
            # the weld carries the part up and it floats in mid-air.
            return place_steps(
                job.component,
                spec,
                lambda: self.site_pose(site_id),
                job.attach_asm,
                seat_on_release=True,
            )
        if stage.name == "press":
            arm = self.arms[stage.arm]
            if self.tool_mgr.state["current_tool"] != "gripper_press":
                raise RuntimeError("press requires gripper_press tool")
            if "press" not in arm.tool_offsets:
                raise RuntimeError(f"{stage.arm} has no press tool site")
            pos, _ = self.objects.perceive(job.component)
            _, slot_yaw = self.site_pose(self.slot_site_ids[job.slot])
            top = pos + np.asarray([0.0, 0.0, float(spec.half_height)])
            return press_steps(
                job.component,
                top,
                slot_yaw,
                arm.tool_offsets["press"],
                (job.component, job.slot),
                fixture_after=True,
            )
        if stage.name == "screw":
            arm = self.arms[stage.arm]
            if self.tool_mgr.state["current_tool"] != "screwdriver":
                raise RuntimeError("screw requires screwdriver tool")
            if "screwdriver" not in arm.tool_offsets:
                raise RuntimeError(f"{stage.arm} has no screwdriver tool site")
            print(f"[Arm2] Tighten {stage.screw}", flush=True)
            pos, yaw = self.site_pose(self.objects.screw_site_ids[stage.screw])
            return screw_steps(pos, yaw, arm.tool_offsets["screwdriver"])
        if stage.name == "inspect":
            slot_pos, slot_yaw = self.site_pose(self.slot_site_ids[job.slot])
            return inspect_steps(slot_pos, slot_yaw)
        raise ValueError(f"unknown stage: {stage.name}")

    # -- scheduling ----------------------------------------------------------
    def stage_ready(self, job: Job, stage: Stage) -> bool:
        arm = self.arms[stage.arm]
        if self.arm_job[stage.arm] is not None:
            return False
        if arm.busy and not arm.parking:
            return False
        if arm.held_component and arm.held_component != job.component:
            return False
        if stage.name == "feed_place":
            site_name = TYPE_STAGING_SITE[self.objects.spec_of(job.component).type_name]
            owner = self.staging_owner[site_name]
            if owner and owner != job.job_id:
                return False
        if stage.name == "assemble_pick":
            if not self.objects.buffer_ready_for(job.component):
                return False
        for token in stage.tokens:
            if self.zone_owner[token] and self.zone_owner[token] != stage.arm:
                return False
        return True

    def start_stage(self, job: Job, stage: Stage) -> None:
        arm = self.arms[stage.arm]
        if arm.parking:
            arm.cancel()
        arm.manual_hold = False
        steps = self.build_stage_steps(job, stage)
        if not steps:
            stage.status = "done"
            self.finish_stage(job, stage)
            return
        for token in stage.tokens:
            self.zone_owner[token] = stage.arm
        stage.status = "active"
        self.arm_job[stage.arm] = job
        arm.start_steps(steps, f"{job.job_id}:{stage.name}")
        self.status = f"{stage.arm} starts {stage.name} ({job.component} -> {job.slot})"

    def finish_stage(self, job: Job, stage: Stage) -> None:
        stage.status = "done"
        for token in stage.tokens:
            if self.zone_owner[token] == stage.arm:
                self.zone_owner[token] = ""
        self.arm_job[stage.arm] = None

        if stage.name == "feed_place":
            self.objects.set_status(job.component, "staged")
            self.objects.mark_buffer_staged(job.component)
        elif stage.name == "assemble_pick":
            if job.staging_site:
                self.staging_owner[job.staging_site] = ""
                job.staging_site = ""
            self.objects.mark_buffer_picked(job.component)
            print("[Arm2] Component grasped", flush=True)
        elif stage.name == "assemble_place":
            self.objects.set_status(job.component, "placed")
            print(f"[Arm2] Insert complete", flush=True)
        elif stage.name == "press":
            self.objects.set_status(job.component, "assembled")
            print(f"[Arm2] Press-fit complete", flush=True)
        elif stage.name == "return_tool" and stage.tool == "screwdriver":
            self.objects.set_status(job.component, "waiting_for_inspection")
            print("[Arm2] Assembly complete, waiting for Arm3 inspection", flush=True)
        elif stage.name == "screw":
            if stage.screw in self.fault_screws:
                self.fault_screws.discard(stage.screw)
                self.objects.set_screw_state(stage.screw, "loose")
                self.status = f"screw {stage.screw} fastening FAILED (injected fault)"
            else:
                self.objects.set_screw_state(stage.screw, "tightened")
                print(f"[Arm2] {stage.screw} tightened", flush=True)
        elif stage.name == "inspect":
            self.evaluate_inspection(job)

        job.stage_idx += 1

    def press_seat(self, component: str, slot: str) -> None:
        """Positioning-fixture behavior of the press-fit: if the part sits
        within the slot capture range, the chamfered fixture guides it onto
        the exact slot pose while the press head holds it down."""
        spec = self.objects.spec_of(component)
        pos, yaw = self.objects.perceive(component)
        slot_pos, slot_yaw = self.site_pose(self.slot_site_ids[slot])
        if float(np.linalg.norm(pos[:2] - slot_pos[:2])) > 0.02:
            return  # too far off: pressing cannot capture it
        seated_yaw = nearest_symmetric_yaw(yaw, slot_yaw, spec.yaw_symmetry_rad)
        seated = np.asarray([slot_pos[0], slot_pos[1], BOARD_TOP_Z + float(spec.half_height) + 0.0005])
        self.objects.teleport(component, seated, seated_yaw)

    def evaluate_inspection(self, job: Job) -> None:
        spec = self.objects.spec_of(job.component)
        obj_pos, obj_yaw = self.objects.perceive(job.component)
        slot_pos, slot_yaw = self.site_pose(self.slot_site_ids[job.slot])
        xy_err = float(np.linalg.norm(obj_pos[:2] - slot_pos[:2]))
        yaw_err = yaw_error_with_symmetry(obj_yaw, slot_yaw, spec.yaw_symmetry_rad)
        loose = [s for s in job.screws if self.objects.screw_state.get(s) != "tightened"]
        misaligned = xy_err >= INSPECT_XY_TOL or math.degrees(yaw_err) >= INSPECT_YAW_TOL_DEG
        ok = not loose and not misaligned

        reasons = []
        if misaligned:
            reasons.append("misaligned")
        if loose:
            reasons.append("loose_screws")
        detail = f"xy={xy_err * 1000:.1f}mm yaw={math.degrees(yaw_err):.1f}deg loose={loose or '-'}"
        result = "pass" if ok else "fail"
        job.inspect_result = result
        self.objects.set_status(job.component, result)
        with self.shared.lock:
            self.shared.inspect_results.append(
                {"job": job.job_id, "slot": job.slot, "result": result, "reason": "+".join(reasons), "detail": detail}
            )
        if ok:
            self.status = f"inspection PASS: {job.component} @ {job.slot} ({detail})"
            return

        # Closed loop (plan 5.4): rework stages chosen by failure reason.
        job.rework_count += 1
        self.status = f"inspection FAIL ({'+'.join(reasons)}): {job.component} @ {job.slot} ({detail}) -> rework #{job.rework_count}"
        if job.rework_count > MAX_REWORK_ROUNDS:
            return
        insert_at = job.stage_idx + 1
        rework: list[Stage] = []
        if misaligned:
            rework.append(Stage("get_tool", "arm2", tool="gripper_press"))
            rework.append(Stage("press", "arm2", tokens=(ZONE_CENTRAL,)))
            rework.append(Stage("return_tool", "arm2", tool="gripper_press"))
        if loose:
            rework.append(Stage("get_tool", "arm2", tool="screwdriver"))
            for screw in loose:
                rework.append(Stage("screw", "arm2", tokens=(ZONE_CENTRAL,), screw=screw))
            rework.append(Stage("return_tool", "arm2", tool="screwdriver"))
        rework.append(Stage("inspect", "arm3", tokens=(ZONE_CENTRAL,)))
        job.stages[insert_at:insert_at] = rework

    # -- main scheduler tick -----------------------------------------------------
    def tick(self, now: float) -> None:
        def on_action(arm: ArmController, step: SkillStep) -> None:
            if step.action == "close_gripper":
                arm.gripper_ctrl = gripper_close_ctrl_for_width(float(step.action_arg))
            elif step.action == "open_gripper":
                arm.gripper_ctrl = GRIPPER_OPEN_CTRL
            elif step.action == "attach":
                instance, record = step.action_arg
                record.update(arm.attach_component(instance))
            elif step.action == "release":
                arm.detach_component(str(step.action_arg))
                arm.gripper_ctrl = GRIPPER_OPEN_CTRL
            elif step.action == "press_seat":
                component, slot = step.action_arg
                self.press_seat(component, slot)
            elif step.action == "tool_dock":
                self.tool_mgr.dock(str(step.action_arg))
            elif step.action == "tool_undock":
                self.tool_mgr.undock(str(step.action_arg))
            elif step.action == "fixture_hold":
                instance = str(step.action_arg)
                self.objects.fixture_hold(instance)
                print("[Arm2] Component fixed by board fixture", flush=True)

        # 1. Advance running skills; finish stages.
        for name, arm in self.arms.items():
            finished = arm.skill_tick(now, on_action)
            if finished:
                job = self.arm_job[name]
                if job is not None:
                    stage = job.current_stage()
                    if stage is not None and stage.status == "active":
                        self.finish_stage(job, stage)

        # 2. Dispatch ready stages (FIFO over jobs -> natural pipelining).
        for job in self.jobs:
            stage = job.current_stage()
            if stage is None or stage.status != "pending":
                continue
            if self.stage_ready(job, stage):
                try:
                    self.start_stage(job, stage)
                except Exception as exc:
                    self.status = f"stage start failed ({job.job_id}:{stage.name}): {exc}"

        # 3. Park idle arms so they never loiter over shared areas.
        for name, arm in self.arms.items():
            if (
                self.arm_job[name] is None
                and not arm.busy
                and not arm.held_component
                and not arm.manual_hold
                and arm.dist_to_home() > 0.05
            ):
                arm.start_park()

        # 4. Order completion bookkeeping.
        if self.jobs and not self.order_done and all(job.done for job in self.jobs):
            self.order_done = True
            elapsed = time.time() - self.order_started
            results = [job.inspect_result or "-" for job in self.jobs]
            self.status = f"order {self.order_id} COMPLETE in {elapsed:.1f}s (inspect: {', '.join(results)})"

    def publish(self) -> None:
        arms_view = {}
        for name, arm in self.arms.items():
            job = self.arm_job[name]
            stage = job.current_stage() if job is not None else None
            arms_view[name] = {
                "stage": stage.name if stage is not None and stage.status == "active" else ("busy" if arm.busy else ""),
                "job": f"{job.component}->{job.slot}" if job is not None else "",
                "step": arm.current_step_label(),
                "held": arm.held_component,
                "pos_err": round(arm.pos_error(), 5),
            }
        jobs_view = []
        for job in self.jobs:
            jobs_view.append(
                {
                    "job": job.job_id,
                    "component": job.component,
                    "slot": job.slot,
                    "progress": f"{job.stage_idx}/{len(job.stages)}" + (" done" if job.done else ""),
                    "inspect": job.inspect_result,
                }
            )
        with self.shared.lock:
            self.shared.arms = arms_view
            self.shared.zone_owners = dict(self.zone_owner)
            self.shared.jobs = jobs_view
            self.shared.screw_states = dict(self.objects.screw_state)
            self.shared.component_states = dict(self.objects.component_status)
            self.shared.order_id = self.order_id
            if self.order_id and not self.order_done:
                self.shared.order_elapsed = time.time() - self.order_started
            self.shared.fault_screws = sorted(self.fault_screws)
            self.shared.status = self.status
            self.shared.server_time = time.time()


# ---------------------------------------------------------------------------
# Main controller loop.
# ---------------------------------------------------------------------------
def run_controller(args: argparse.Namespace) -> int:
    import mujoco
    import mujoco.viewer

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"找不到 XML 文件: {xml_path}")

    shared = SharedState()
    server = start_http_server(shared, "127.0.0.1", int(args.port))
    actual_port = int(server.server_address[1])
    print(f"[1/5] HTTP control server: http://127.0.0.1:{actual_port}", flush=True)

    if not args.no_terminal_commands:
        start_terminal_thread(shared)

    ui_proc: Optional[subprocess.Popen[Any]] = None
    if not args.no_ui:
        ui_python = find_python_for_ui(args.ui_python)
        cmd = [ui_python, str(Path(__file__).resolve()), "--ui-client", "--port", str(actual_port)]
        print("[UI] " + " ".join(cmd), flush=True)
        ui_proc = subprocess.Popen(cmd)

    print(f"[2/5] Loading XML: {xml_path}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    original_geom_rgba = model.geom_rgba.copy()
    original_site_rgba = model.site_rgba.copy()
    model.opt.timestep = float(args.dt)
    if args.zero_gravity:
        model.opt.gravity[:] = 0.0

    # Gravity compensation for all three robot subtrees.
    model.body_gravcomp[:] = 0.0
    if args.gravity_comp:
        for name in ARM_NAMES:
            root_id = int(model.body(f"{name}_base").id)
            for body_id in range(model.nbody):
                b = body_id
                while b != 0:
                    if b == root_id:
                        model.body_gravcomp[body_id] = 1.0
                        break
                    b = int(model.body_parentid[b])

    key_id = int(model.key("home").id)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    objects = ObjectManager(model, data)
    scatter = objects.randomize_rack_poses(args.seed)
    print(f"[scatter] seed={args.seed}: " + "  ".join(f"{k}=({v[0]:.2f},{v[1]:.2f},{math.degrees(v[2]):.0f}°)" for k, v in scatter.items()), flush=True)

    arms = {name: ArmController(model, data, name, args) for name in ARM_NAMES}
    for arm in arms.values():
        arm.reset_hold()
        data.ctrl[arm.gripper_act_id] = GRIPPER_OPEN_CTRL

    world = PipelineWorld(model, data, arms, objects, shared, args)

    # Visibility helpers.
    ee_geom_ids: list[int] = []
    ee_site_ids: list[int] = []
    mocap_geom_ids: list[int] = []
    mocap_site_ids: list[int] = []
    for name in ARM_NAMES:
        for axis in ("x", "y", "z"):
            try:
                ee_geom_ids.append(int(model.geom(f"{name}_ee_axis_{axis}").id))
            except KeyError:
                pass
        ee_site_ids.append(int(model.site(f"{name}_attachment_site").id))
        target_body_id = int(model.body(f"{name}_target").id)
        mocap_geom_ids.extend(int(i) for i in range(model.ngeom) if int(model.geom_bodyid[i]) == target_body_id)
        mocap_site_ids.extend(int(i) for i in range(model.nsite) if int(model.site_bodyid[i]) == target_body_id)

    def apply_visibility(geom_ids: list[int], site_ids: list[int], visible: bool) -> None:
        for gid in geom_ids:
            model.geom_rgba[gid] = original_geom_rgba[gid]
            if not visible:
                model.geom_rgba[gid, 3] = 0.0
        for sid in site_ids:
            model.site_rgba[sid] = original_site_rgba[sid]
            if not visible:
                model.site_rgba[sid, 3] = 0.0

    print("[3/5] Opening MuJoCo viewer.", flush=True)
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=args.show_mujoco_ui,
        show_right_ui=args.show_mujoco_ui,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.lookat[:] = [0.0, 0.35, 0.2]
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -35
        print("[4/5] Three-arm pipeline controller running.", flush=True)

        running = True
        last_state_time = 0.0

        while viewer.is_running() and running:
            step_start = time.time()

            while True:
                try:
                    command = shared.commands.get_nowait()
                except queue.Empty:
                    break
                try:
                    ctype = command.get("type")
                    if ctype == "order":
                        world.load_order(dict(command.get("order", {})))
                    elif ctype == "stop":
                        world.stop_all()
                    elif ctype == "fault":
                        world.inject_fault(str(command.get("screw", "")), str(command.get("component", "")))
                    elif ctype == "scatter":
                        if any(not job.done for job in world.jobs):
                            world.status = "scatter refused: order is running"
                        else:
                            seed = command.get("seed", None)
                            result = objects.randomize_rack_poses(seed)
                            objects.reset_statuses()
                            world.status = f"rack re-scattered ({len(result)} components)"
                    elif ctype == "move":
                        arm_name = str(command.get("arm", ""))
                        waypoints = [pose_array_to_command(item) for item in command.get("waypoints", [])]
                        if arm_name in arms and waypoints and world.arm_job[arm_name] is None:
                            if arms[arm_name].parking:
                                arms[arm_name].cancel()
                            arms[arm_name].set_mocap_pose(waypoints[0])
                            arms[arm_name].manual_hold = True
                            world.status = f"{arm_name} manual move"
                    elif ctype == "quit":
                        running = False
                except Exception as exc:
                    world.status = f"command failed: {exc}"

            apply_visibility(ee_geom_ids, ee_site_ids, bool(shared.show_ee_axes))
            apply_visibility(mocap_geom_ids, mocap_site_ids, bool(shared.show_mocap))

            now = time.time()
            world.tick(now)

            for arm in arms.values():
                arm.control_tick()

            mujoco.mj_step(model, data)
            viewer.sync()

            if now - last_state_time > 0.1:
                world.publish()
                last_state_time = now

            sleep_time = float(args.dt) - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    shared.update(status="viewer closed", viewer_running=False)
    server.shutdown()
    if ui_proc is not None and ui_proc.poll() is None:
        ui_proc.terminate()
    print("[5/5] Shutdown complete.", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-FR3 flexible production line: feed -> assemble -> press -> screw -> inspect/rework")
    parser.add_argument("--ui-client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--ui-python", type=str, default=None)
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--no-terminal-commands", action="store_true")
    parser.add_argument("--xml", type=str, default=str(Path(__file__).with_name("multi_arm_line.xml")))
    parser.add_argument("--seed", type=int, default=None, help="Rack scatter seed (default random)")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--kpos", type=float, default=3.6)
    parser.add_argument("--kori", type=float, default=3.6)
    parser.add_argument("--joint-centering-gain", type=float, default=2.5)
    parser.add_argument("--joint-limit-barrier-gain", type=float, default=0.04)
    parser.add_argument("--max-nullspace-speed", type=float, default=4.0)
    parser.add_argument("--joint-secondary-gate", type=float, default=0.25)
    parser.add_argument("--arm-deadband-pos", type=float, default=2e-4)
    parser.add_argument("--arm-deadband-ori", type=float, default=2e-3)
    parser.add_argument("--wrist-spin-speed-deg-s", type=float, default=360.0)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--max-angvel", type=float, default=21.0)
    parser.add_argument("--position-tolerance", type=float, default=0.003)
    parser.add_argument("--orientation-tolerance", type=float, default=0.03)
    parser.add_argument("--zero-gravity", action="store_true")
    parser.add_argument("--gravity-comp", action="store_true", default=True)
    parser.add_argument("--no-gravity-comp", action="store_false", dest="gravity_comp", help=argparse.SUPPRESS)
    parser.add_argument("--show-mujoco-ui", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ui_client:
        return run_ui_client(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
