"""FR3 differential IK controller for an ellipsoid + rotating wheel scene.

Run from your project root with mjpython, for example:

    mjpython grasp_ellipsoid_wheel_control.py \
        --xml franka_fr3/grasp_ellipsoid_wheel_scene.xml

Why subprocess UI?
------------------
On macOS, `mjpython` is usually the safest launcher for MuJoCo's native viewer,
but Qt/PySide6 may hang when QApplication is created inside `mjpython`.  This
script therefore runs:

  1. MuJoCo simulation + viewer in the main `mjpython` process.
  2. PySide6 UI in a normal Python subprocess.

The two processes communicate through localhost HTTP.

UI features:
  - current EE pose display: x y z roll pitch yaw
  - workpiece frame world pose display and direct 6D pose reset
  - move one 6D pose
  - xyz-function trajectory: x(t), y(t), z(t), t_start, t_end, samples
  - direct function trajectory is time-streamed so the target visibly moves along the curve
  - function_path first holds the current mocap pose before streaming
  - gripper slider
  - show/hide EE axes and mocap/target axes
  - show/clear mocap center trajectory trail

Function convention:
  - use variable `t`
  - allowed functions: sin, cos, tan, sqrt, exp, log, pi, etc. from Python math
  - examples:
      x(t) = 0.45 + 0.05*cos(2*pi*t)
      y(t) = 0.00 + 0.05*sin(2*pi*t)
      z(t) = 0.45 + 0.02*t
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
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ----------------------------- Controller params -----------------------------

integration_dt: float = 0.1
damping: float = 1e-4
Kpos: float = 0.95
Kori: float = 0.95
gravity_compensation: bool = True
dt: float = 0.002
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=float)
max_angvel = 0.785
align_step1_fixed_pose_rad = np.asarray([0.623, 0.0, 0.309, math.pi, 0.0, 0.0], dtype=float)
align_step1_fixed_pose_deg_text = "0.623000 0 0.309000 180.00000 0 0"
surface_scan_samples_per_side = 160
double_sphere_radius = 0.0725
double_sphere_center_half_distance = 0.0525
surface_scan_circle_radius = math.sqrt(
    max(0.0, double_sphere_radius * double_sphere_radius - double_sphere_center_half_distance * double_sphere_center_half_distance)
)
surface_scan_yaw_center = math.pi
surface_scan_step_time = 0.06
surface_scan_transition_lift = 0.040

# Names expected in your FR3+gripper XML.
SITE_NAME = "attachment_site"
MOCAP_NAME = "target"
WORKPIECE_MOCAP_NAME = "workpiece_mocap_target"
WORKPIECE_SITE_NAME = "workpiece_frame"
EE_AXIS_CYLINDER_GEOM_NAME = "ee_axis_z_center_cylinder"
ELLIPSOID_GEOM_NAME = "grasp_ellipsoid_geom"
WHEEL_CYLINDER_GEOM_NAME = "wheel_cylinder"
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
ELLIPSOID_BODY_NAME = "grasp_ellipsoid"


@dataclass(frozen=True)
class PoseCommand:
    position: np.ndarray
    quat: np.ndarray
    raw_pose: np.ndarray  # x y z roll pitch yaw, radians


@dataclass
class MotionState:
    current_goal: Optional[PoseCommand] = None
    remaining_goals: Optional[list[PoseCommand]] = None
    active_goal_index: int = 0
    total_goal_count: int = 0
    active_mode: str = ""

    def clear(self) -> None:
        self.current_goal = None
        self.remaining_goals = []
        self.active_goal_index = 0
        self.total_goal_count = 0
        self.active_mode = ""


@dataclass
class WorkpieceLockState:
    mode: str = "inactive"
    site_to_body_pos: Optional[np.ndarray] = None
    site_to_body_mat: Optional[np.ndarray] = None

    def clear(self) -> None:
        self.mode = "inactive"
        self.site_to_body_pos = None
        self.site_to_body_mat = None

    def start_closing(self) -> None:
        self.mode = "closing"
        self.site_to_body_pos = None
        self.site_to_body_mat = None


@dataclass
class SurfaceScanState:
    active: bool = False
    streaming: bool = False
    index: int = 0
    last_step_at: float = 0.0
    clearance: float = 0.0
    labels: list[str] | None = None

    def clear(self) -> None:
        self.active = False
        self.streaming = False
        self.index = 0
        self.last_step_at = 0.0
        self.clearance = 0.0
        self.labels = None

    def label(self) -> str:
        if not self.labels or self.index < 0 or self.index >= len(self.labels):
            return "inactive"
        return self.labels[self.index]


@dataclass
class RobotConfig:
    xml_path: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    gripper_actuator_name: Optional[str] = "gripper"


ROBOT_CONFIGS = {
    "fr3": RobotConfig(
        xml_path="franka_fr3/grasp_ellipsoid_wheel_scene.xml",
        joint_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
        actuator_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
    ),
    "panda": RobotConfig(
        xml_path="franka_emika_panda/scene.xml",
        joint_names=tuple(f"joint{i}" for i in range(1, 8)),
        actuator_names=tuple(f"joint{i}" for i in range(1, 8)),
        gripper_actuator_name=None,
    ),
}


# ----------------------------- Pose conversion -------------------------------

def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX roll-pitch-yaw to MuJoCo quaternion order [w, x, y, z]."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def rpy_from_mat(xmat: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo 3x3 rotation matrix to ZYX roll-pitch-yaw."""
    r = np.asarray(xmat, dtype=float).reshape(3, 3)
    pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=float)


def pose_array_to_command(values: list[float] | np.ndarray) -> PoseCommand:
    pose = np.asarray(values, dtype=float)
    if pose.shape != (6,):
        raise ValueError("pose must have 6 elements: x y z roll pitch yaw")
    quat = quat_from_rpy(float(pose[3]), float(pose[4]), float(pose[5]))
    return PoseCommand(position=pose[:3].copy(), quat=quat, raw_pose=pose.copy())


def parse_waypoints_rad(waypoints: list[list[float]]) -> list[PoseCommand]:
    return [pose_array_to_command(item) for item in waypoints]


def format_pose(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(x):.6f}" for x in values)


# ----------------------------- HTTP IPC server -------------------------------

class SharedState:
    def __init__(self, initial_gripper: float) -> None:
        self.lock = threading.Lock()
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self.pose_rad = [0.0] * 6
        self.pose_deg = [0.0] * 6
        self.pos_err = 0.0
        self.ori_err = 0.0
        self.status = "正在启动 MuJoCo..."
        self.gripper_value = float(initial_gripper)
        self.gripper_command_value = float(initial_gripper)
        self.gripper_actual_value = float(initial_gripper)
        self.gripper_blocked = False
        self.gripper_close_limit = 0.0
        self.gripper_auto_closing = False
        self.show_ee_axes = True
        self.show_mocap = True
        self.show_trail = False
        self.mocap_z_comp = 0.0
        self.has_gripper = True
        self.viewer_running = False
        self.waypoint_index = 0
        self.waypoint_count = 0
        self.active_mode = "idle"
        self.trail_point_count = 0
        self.workpiece_pose_rad: Optional[list[float]] = None
        self.workpiece_pose_deg: Optional[list[float]] = None
        self.align_step1_error_text = f"步骤1固定目标: {align_step1_fixed_pose_deg_text}"
        self.workpiece_lock_status = "inactive"
        self.scan_status = "inactive"
        self.server_time = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "pose_rad": list(self.pose_rad),
                "pose_deg": list(self.pose_deg),
                "pos_err": float(self.pos_err),
                "ori_err": float(self.ori_err),
                "status": str(self.status),
                "gripper": float(self.gripper_value),
                "gripper_command": float(self.gripper_command_value),
                "gripper_actual": float(self.gripper_actual_value),
                "gripper_blocked": bool(self.gripper_blocked),
                "gripper_close_limit": float(self.gripper_close_limit),
                "gripper_auto_closing": bool(self.gripper_auto_closing),
                "show_ee_axes": bool(self.show_ee_axes),
                "show_mocap": bool(self.show_mocap),
                "show_trail": bool(self.show_trail),
                "mocap_z_comp": float(self.mocap_z_comp),
                "has_gripper": bool(self.has_gripper),
                "viewer_running": bool(self.viewer_running),
                "waypoint_index": int(self.waypoint_index),
                "waypoint_count": int(self.waypoint_count),
                "active_mode": str(self.active_mode),
                "trail_point_count": int(self.trail_point_count),
                "workpiece_pose_rad": None if self.workpiece_pose_rad is None else list(self.workpiece_pose_rad),
                "workpiece_pose_deg": None if self.workpiece_pose_deg is None else list(self.workpiece_pose_deg),
                "align_step1_error_text": str(self.align_step1_error_text),
                "workpiece_lock_status": str(self.workpiece_lock_status),
                "scan_status": str(self.scan_status),
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
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(raw or "{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON payload must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/state":
            self._send_json(self.shared.snapshot())
        else:
            self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/move":
                waypoints = payload.get("waypoints", [])
                mode = str(payload.get("mode", "move"))
                if not isinstance(waypoints, list) or len(waypoints) == 0:
                    raise ValueError("waypoints must be a non-empty list")
                if mode not in {"move_one_step", "function_path"}:
                    raise ValueError("mode must be move_one_step or function_path")
                if mode == "move_one_step" and len(waypoints) != 1:
                    raise ValueError("move_one_step requires exactly one waypoint")
                self.shared.commands.put({"type": "move", "mode": mode, "waypoints": waypoints})
                self._send_json({"ok": True})
            elif self.path == "/stop":
                self.shared.commands.put({"type": "stop"})
                self._send_json({"ok": True})
            elif self.path == "/gripper":
                value = max(0.0, min(255.0, float(payload.get("value", 255.0))))
                self.shared.update(gripper_command_value=value, status=f"gripper = {value:.0f}")
                self.shared.commands.put({"type": "gripper", "value": value})
                self._send_json({"ok": True, "value": value})
            elif self.path == "/gripper_open":
                self.shared.update(
                    gripper_command_value=255.0,
                    gripper_value=255.0,
                    gripper_blocked=False,
                    gripper_close_limit=0.0,
                    gripper_auto_closing=False,
                    workpiece_lock_status="inactive",
                    status="gripper open",
                )
                self.shared.commands.put({"type": "gripper_open"})
                self._send_json({"ok": True})
            elif self.path == "/gripper_close":
                self.shared.update(
                    gripper_command_value=0.0,
                    gripper_auto_closing=True,
                    status="gripper close：关闭直到受阻或完全闭合",
                )
                self.shared.commands.put({"type": "gripper_close"})
                self._send_json({"ok": True})
            elif self.path == "/ee_axes":
                visible = bool(payload.get("visible", True))
                self.shared.update(show_ee_axes=visible, status="显示机械臂末端坐标轴" if visible else "隐藏机械臂末端坐标轴")
                self.shared.commands.put({"type": "ee_axes", "visible": visible})
                self._send_json({"ok": True, "visible": visible})
            elif self.path == "/mocap":
                visible = bool(payload.get("visible", True))
                self.shared.update(show_mocap=visible, status="显示 mocap/目标坐标轴" if visible else "隐藏 mocap/目标坐标轴")
                self.shared.commands.put({"type": "mocap", "visible": visible})
                self._send_json({"ok": True, "visible": visible})
            elif self.path == "/trail":
                enabled = bool(payload.get("enabled", True))
                self.shared.update(show_trail=enabled, status="开始记录 mocap 中心轨迹" if enabled else "停止记录 mocap 中心轨迹")
                self.shared.commands.put({"type": "trail", "enabled": enabled})
                self._send_json({"ok": True, "enabled": enabled})
            elif self.path == "/clear_trail":
                self.shared.commands.put({"type": "clear_trail"})
                self.shared.update(trail_point_count=0, status="已清除 mocap 中心轨迹")
                self._send_json({"ok": True})
            elif self.path == "/offset":
                value = float(payload.get("mocap_z_comp", 0.0))
                self.shared.update(mocap_z_comp=value, status=f"mocap z 补偿 = {value:.4f} m")
                self.shared.commands.put({"type": "offset", "mocap_z_comp": value})
                self._send_json({"ok": True, "mocap_z_comp": value})
            elif self.path == "/workpiece_pose":
                with self.shared.lock:
                    lock_status = str(self.shared.workpiece_lock_status)
                if lock_status == "closing":
                    raise ValueError("待磨件正在锁定；请等待步骤2完成后再移动待磨件。")
                pose = payload.get("pose", [])
                if not isinstance(pose, list) or len(pose) != 6:
                    raise ValueError("pose must have 6 elements: x y z roll pitch yaw")
                values = [float(x) for x in pose]
                self.shared.commands.put({"type": "workpiece_pose", "pose": values})
                if lock_status == "locked":
                    self.shared.update(status=f"请求移动 locked 待磨件/机械臂: {format_pose(values)}")
                else:
                    self.shared.update(status=f"请求移动待磨件: {format_pose(values)}")
                self._send_json({"ok": True, "pose": values})
            elif self.path == "/align_workpiece_handle_step1":
                self.shared.commands.put({"type": "align_workpiece_handle_step1"})
                self.shared.update(status=f"请求执行对齐步骤1：移动到固定目标 {align_step1_fixed_pose_deg_text}")
                self._send_json({"ok": True})
            elif self.path == "/align_workpiece_handle_step2":
                self.shared.commands.put({"type": "align_workpiece_handle_step2"})
                self.shared.update(status="请求执行对齐步骤2：关闭 gripper 并锁定待磨件", workpiece_lock_status="closing")
                self._send_json({"ok": True})
            elif self.path == "/ellipsoid_surface_scan":
                with self.shared.lock:
                    lock_status = str(self.shared.workpiece_lock_status)
                if lock_status != "locked":
                    raise ValueError("待磨件尚未 locked；请先执行步骤2。")
                clearance = float(payload.get("surface_clearance", 0.0))
                clearance = float(np.clip(clearance, -0.020, 0.020))
                self.shared.commands.put({"type": "ellipsoid_surface_scan", "surface_clearance": clearance})
                self.shared.update(
                    status=f"请求执行圆周扫描：surface_clearance={clearance:.4f}m",
                    scan_status=f"waiting_start, clearance={clearance:.4f}m",
                )
                self._send_json({"ok": True, "surface_clearance": clearance})
            elif self.path == "/quit":
                self.shared.commands.put({"type": "quit"})
                self._send_json({"ok": True})
            else:
                self._send_json({"ok": False, "error": "not found"}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)


def start_http_server(shared: SharedState, host: str, port: int) -> ThreadingHTTPServer:
    RequestHandler.shared = shared
    server = ThreadingHTTPServer((host, port), RequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def get_json(url: str, timeout: float = 0.7) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: float = 0.7) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ----------------------------- UI process ------------------------------------

def find_python_for_ui(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        p = Path(conda_prefix) / "bin" / "python"
        if p.exists():
            return str(p)
    if "mjpython" not in Path(sys.executable).name.lower():
        return sys.executable
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found and "mjpython" not in Path(found).name.lower():
            return found
    return sys.executable


def launch_ui_subprocess(script_path: Path, port: int, ui_python: str) -> subprocess.Popen[Any]:
    cmd = [ui_python, str(script_path), "--ui-client", "--port", str(port)]
    print("[UI] 启动 PySide6 子进程：" + " ".join(cmd), flush=True)
    return subprocess.Popen(cmd)


# ----------------------------- Safe function eval ----------------------------

SAFE_MATH_NAMES: dict[str, Any] = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
SAFE_MATH_NAMES.update({"abs": abs, "min": min, "max": max, "pow": pow})


def normalize_expr(expr: str) -> str:
    expr = expr.strip()
    # Let the user type either "0.45 + ..." or "x(t)=0.45 + ...".
    if "=" in expr:
        expr = expr.split("=", 1)[1].strip()
    return expr


def safe_eval_expr(expr: str, t: float) -> float:
    local = dict(SAFE_MATH_NAMES)
    local["t"] = float(t)
    return float(eval(normalize_expr(expr), {"__builtins__": {}}, local))


# ----------------------------- PySide6 client --------------------------------

def run_ui_client(args: argparse.Namespace) -> int:
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSlider,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        print("没有找到 PySide6。请先运行：pip install PySide6", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    class ControlPanel(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.base_url = f"http://127.0.0.1:{args.port}"
            self.last_pose_rad = [0.0] * 6
            self.last_workpiece_pose_rad: Optional[list[float]] = None
            self.last_workpiece_pose_deg: Optional[list[float]] = None
            self.gripper_blocked = False
            self.gripper_close_limit = 0
            self.default_unit = "rad"
            self._build()

        def _build(self) -> None:
            self.setWindowTitle("FR3 PySide6 控制面板 - Ellipsoid + Wheel")
            self.resize(820, 840)
            self.setMinimumSize(740, 760)
            self.setStyleSheet(
                """
                QWidget { font-size: 13px; }
                QGroupBox { font-weight: bold; margin-top: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
                QLabel#TitleLabel { font-size: 18px; font-weight: bold; }
                QLabel#MonoLabel { font-family: Menlo, Monaco, Consolas, monospace; font-size: 12px; }
                QPushButton { padding: 4px 10px; }
                """
            )
            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(8)

            title = QLabel("FR3 末端位姿 / 待磨件位姿 / 函数轨迹 / Gripper 控制")
            title.setObjectName("TitleLabel")
            layout.addWidget(title)

            # Current pose.
            pose_group = QGroupBox("当前末端六维位姿")
            pose_layout = QVBoxLayout(pose_group)
            pose_layout.addWidget(QLabel("rad:  x y z roll pitch yaw"))
            self.pose_rad_label = QLabel("等待仿真数据...")
            self.pose_rad_label.setObjectName("MonoLabel")
            pose_layout.addWidget(self.pose_rad_label)
            pose_layout.addWidget(QLabel("deg:  x y z roll pitch yaw"))
            self.pose_deg_label = QLabel("等待仿真数据...")
            self.pose_deg_label.setObjectName("MonoLabel")
            pose_layout.addWidget(self.pose_deg_label)
            self.err_label = QLabel("pos_err: --   ori_err: --")
            pose_layout.addWidget(self.err_label)
            layout.addWidget(pose_group)

            # Workpiece frame pose.
            workpiece_group = QGroupBox("待磨件六维坐标")
            workpiece_layout = QVBoxLayout(workpiece_group)
            workpiece_layout.addWidget(QLabel("rad:  x y z roll pitch yaw"))
            self.workpiece_pose_rad_label = QLabel("等待仿真数据...")
            self.workpiece_pose_rad_label.setObjectName("MonoLabel")
            workpiece_layout.addWidget(self.workpiece_pose_rad_label)
            workpiece_layout.addWidget(QLabel("deg:  x y z roll pitch yaw"))
            self.workpiece_pose_deg_label = QLabel("等待仿真数据...")
            self.workpiece_pose_deg_label.setObjectName("MonoLabel")
            workpiece_layout.addWidget(self.workpiece_pose_deg_label)

            workpiece_row = QHBoxLayout()
            workpiece_row.addWidget(QLabel("输入姿态角单位："))
            self.workpiece_unit_combo = QComboBox()
            self.workpiece_unit_combo.addItems(["rad", "deg"])
            workpiece_row.addWidget(self.workpiece_unit_combo)
            workpiece_row.addStretch(1)
            workpiece_layout.addLayout(workpiece_row)

            self.workpiece_pose_boxes: dict[str, QDoubleSpinBox] = {}
            workpiece_grid = QGridLayout()
            workpiece_defaults = {"x": 0.62, "y": 0.0, "z": 0.3305, "roll": 3.1416, "pitch": 0.0, "yaw": 0.0}
            for col, name in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
                box = QDoubleSpinBox()
                box.setDecimals(6)
                box.setRange(-10000, 10000)
                box.setSingleStep(0.01 if name in {"x", "y", "z"} else 0.05)
                box.setValue(workpiece_defaults[name])
                self.workpiece_pose_boxes[name] = box
                workpiece_grid.addWidget(QLabel(name), 0, col)
                workpiece_grid.addWidget(box, 1, col)
            workpiece_layout.addLayout(workpiece_grid)

            workpiece_btns = QHBoxLayout()
            fill_workpiece_btn = QPushButton("将当前位姿填入")
            fill_workpiece_btn.clicked.connect(self.fill_current_workpiece_pose)
            fill_workpiece_move_btn = QPushButton("填入Move框")
            fill_workpiece_move_btn.clicked.connect(self.fill_workpiece_pose_to_move_box)
            move_workpiece_btn = QPushButton("移动待磨件/机械臂到该位姿")
            move_workpiece_btn.clicked.connect(self.move_workpiece_pose)
            workpiece_btns.addWidget(fill_workpiece_btn)
            workpiece_btns.addWidget(fill_workpiece_move_btn)
            workpiece_btns.addWidget(move_workpiece_btn)
            workpiece_btns.addStretch(1)
            workpiece_layout.addLayout(workpiece_btns)
            layout.addWidget(workpiece_group)

            align_group = QGroupBox("待磨件把手对齐")
            align_layout = QVBoxLayout(align_group)
            align_btns = QHBoxLayout()
            align_step1_btn = QPushButton("执行对齐步骤1")
            align_step1_btn.clicked.connect(self.align_workpiece_handle_step1)
            align_step2_btn = QPushButton("执行对齐步骤2")
            align_step2_btn.clicked.connect(self.align_workpiece_handle_step2)
            scan_btn = QPushButton("执行圆周扫描")
            scan_btn.clicked.connect(self.execute_ellipsoid_surface_scan)
            align_btns.addWidget(align_step1_btn)
            align_btns.addWidget(align_step2_btn)
            align_btns.addWidget(scan_btn)
            align_btns.addStretch(1)
            align_layout.addLayout(align_btns)
            clearance_row = QHBoxLayout()
            clearance_row.addWidget(QLabel("surface_clearance(m):"))
            self.surface_clearance_box = QDoubleSpinBox()
            self.surface_clearance_box.setDecimals(6)
            self.surface_clearance_box.setRange(-0.020, 0.020)
            self.surface_clearance_box.setSingleStep(0.0005)
            self.surface_clearance_box.setValue(0.0)
            clearance_row.addWidget(self.surface_clearance_box)
            clearance_row.addStretch(1)
            align_layout.addLayout(clearance_row)
            self.align_step1_error_label = QLabel(f"步骤1固定目标: {align_step1_fixed_pose_deg_text}")
            self.align_step1_error_label.setObjectName("MonoLabel")
            self.align_step1_error_label.setWordWrap(True)
            align_layout.addWidget(self.align_step1_error_label)
            self.workpiece_lock_label = QLabel("待磨件锁定: inactive")
            self.workpiece_lock_label.setObjectName("MonoLabel")
            align_layout.addWidget(self.workpiece_lock_label)
            self.scan_status_label = QLabel("扫描状态: inactive")
            self.scan_status_label.setObjectName("MonoLabel")
            self.scan_status_label.setWordWrap(True)
            align_layout.addWidget(self.scan_status_label)
            layout.addWidget(align_group)

            # Move one step.
            one_group = QGroupBox("Move One Step")
            one_layout = QVBoxLayout(one_group)
            unit_row = QHBoxLayout()
            unit_row.addWidget(QLabel("姿态角单位："))
            self.unit_combo = QComboBox()
            self.unit_combo.addItems(["rad", "deg"])
            unit_row.addWidget(self.unit_combo)
            unit_row.addStretch(1)
            one_layout.addLayout(unit_row)

            self.pose_boxes: dict[str, QDoubleSpinBox] = {}
            grid = QGridLayout()
            defaults = {"x": 0.45, "y": 0.0, "z": 0.45, "roll": 3.1416, "pitch": 0.0, "yaw": 0.0}
            for col, name in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
                box = QDoubleSpinBox()
                box.setDecimals(6)
                box.setRange(-10000, 10000)
                box.setSingleStep(0.01 if name in {"x", "y", "z"} else 0.05)
                box.setValue(defaults[name])
                self.pose_boxes[name] = box
                grid.addWidget(QLabel(name), 0, col)
                grid.addWidget(box, 1, col)
            one_layout.addLayout(grid)

            row = QHBoxLayout()
            btn = QPushButton("移动到该位姿")
            btn.clicked.connect(self.move_one_step)
            row.addWidget(btn)
            btn2 = QPushButton("把当前位姿填入")
            btn2.clicked.connect(self.fill_current_pose)
            row.addWidget(btn2)
            stop_btn = QPushButton("停止")
            stop_btn.clicked.connect(self.stop_motion)
            row.addWidget(stop_btn)
            row.addStretch(1)
            one_layout.addLayout(row)
            layout.addWidget(one_group)

            # Function trajectory.
            func_group = QGroupBox("函数轨迹：x(t), y(t), z(t)")
            func_layout = QVBoxLayout(func_group)
            func_layout.addWidget(QLabel("输入只包含 xyz 的参数函数；t 从 t_start 采样到 t_end。姿态默认保持当前末端姿态。"))
            self.expr_x = QLineEdit("0.45 + 0.05*cos(2*pi*t)")
            self.expr_y = QLineEdit("0.00 + 0.05*sin(2*pi*t)")
            self.expr_z = QLineEdit("0.45 + 0.02*t")
            expr_grid = QGridLayout()
            for row_i, (label, widget) in enumerate((("x(t)", self.expr_x), ("y(t)", self.expr_y), ("z(t)", self.expr_z))):
                expr_grid.addWidget(QLabel(label), row_i, 0)
                expr_grid.addWidget(widget, row_i, 1)
            func_layout.addLayout(expr_grid)

            param_row = QHBoxLayout()
            self.t_start = QDoubleSpinBox(); self.t_start.setRange(-1e6, 1e6); self.t_start.setDecimals(6); self.t_start.setValue(0.0)
            self.t_end = QDoubleSpinBox(); self.t_end.setRange(-1e6, 1e6); self.t_end.setDecimals(6); self.t_end.setValue(1.0)
            self.samples = QSpinBox(); self.samples.setRange(2, 2000); self.samples.setValue(60)
            for label, widget in (("t_start", self.t_start), ("t_end", self.t_end), ("采样点数", self.samples)):
                param_row.addWidget(QLabel(label))
                param_row.addWidget(widget)
            param_row.addStretch(1)
            func_layout.addLayout(param_row)

            ori_row = QHBoxLayout()
            self.function_ori_combo = QComboBox()
            self.function_ori_combo.addItems(["保持当前姿态", "使用 Move One Step 的 roll/pitch/yaw"])
            ori_row.addWidget(QLabel("函数轨迹姿态："))
            ori_row.addWidget(self.function_ori_combo)
            ori_row.addStretch(1)
            func_layout.addLayout(ori_row)

            func_btns = QHBoxLayout()
            exec_btn = QPushButton("直接执行函数轨迹")
            exec_btn.clicked.connect(self.execute_function_path)
            sample_btn = QPushButton("填入圆形示例")
            sample_btn.clicked.connect(self.fill_circle_example)
            func_btns.addWidget(exec_btn)
            func_btns.addWidget(sample_btn)
            func_btns.addStretch(1)
            func_layout.addLayout(func_btns)
            layout.addWidget(func_group)

            # Gripper.
            grip_group = QGroupBox("Gripper 开合")
            grip_layout = QVBoxLayout(grip_group)
            self.gripper_slider = QSlider(Qt.Orientation.Horizontal)
            self.gripper_slider.setRange(0, 255)
            self.gripper_slider.setValue(255)
            self.gripper_slider.valueChanged.connect(self.set_gripper)
            grip_layout.addWidget(QLabel("0 = 闭合，255 = 张开"))
            grip_layout.addWidget(self.gripper_slider)
            grip_btns = QHBoxLayout()
            open_btn = QPushButton("open")
            open_btn.clicked.connect(self.open_gripper)
            close_btn = QPushButton("close")
            close_btn.clicked.connect(self.close_gripper)
            self.gripper_state_label = QLabel("actual: --   command: --")
            grip_btns.addWidget(open_btn)
            grip_btns.addWidget(close_btn)
            grip_btns.addWidget(self.gripper_state_label)
            grip_btns.addStretch(1)
            grip_layout.addLayout(grip_btns)
            layout.addWidget(grip_group)

            # Visibility.
            vis_group = QGroupBox("显示 / 隐藏")
            vis_layout = QHBoxLayout(vis_group)
            self.ee_axes_checkbox = QCheckBox("显示机械臂末端坐标轴")
            self.ee_axes_checkbox.setChecked(True)
            self.mocap_checkbox = QCheckBox("显示 mocap 和 mocap 坐标轴")
            self.mocap_checkbox.setChecked(True)
            self.ee_axes_checkbox.toggled.connect(lambda v: self.post("/ee_axes", {"visible": bool(v)}))
            self.mocap_checkbox.toggled.connect(lambda v: self.post("/mocap", {"visible": bool(v)}))
            vis_layout.addWidget(self.ee_axes_checkbox)
            vis_layout.addWidget(self.mocap_checkbox)
            vis_layout.addStretch(1)
            layout.addWidget(vis_group)

            trail_group = QGroupBox("Mocap 中心运动轨迹")
            trail_layout = QHBoxLayout(trail_group)
            self.trail_checkbox = QCheckBox("留下 mocap 中心轨迹")
            self.trail_checkbox.setChecked(False)
            self.trail_checkbox.toggled.connect(lambda v: self.post("/trail", {"enabled": bool(v)}))
            clear_trail_btn = QPushButton("清除运动轨迹")
            clear_trail_btn.clicked.connect(lambda: self.post("/clear_trail", {}))
            self.trail_count_label = QLabel("轨迹点: 0")
            trail_layout.addWidget(self.trail_checkbox)
            trail_layout.addWidget(clear_trail_btn)
            trail_layout.addWidget(self.trail_count_label)
            trail_layout.addStretch(1)
            layout.addWidget(trail_group)

            line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); layout.addWidget(line)
            self.status_label = QLabel("正在连接控制器...")
            self.status_label.setWordWrap(True)
            layout.addWidget(self.status_label)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.poll_state)
            self.timer.start(100)
            scroll.setWidget(content)
            root_layout.addWidget(scroll)

        def post(self, endpoint: str, payload: dict[str, Any]) -> bool:
            try:
                resp = post_json(f"{self.base_url}{endpoint}", payload, timeout=1.0)
                if not resp.get("ok", False):
                    raise RuntimeError(str(resp.get("error", "未知错误")))
                return True
            except Exception as exc:
                QMessageBox.critical(self, "发送命令失败", str(exc))
                return False

        def closeEvent(self, event) -> None:  # type: ignore[override]
            try:
                post_json(f"{self.base_url}/quit", {}, timeout=0.3)
            except Exception:
                pass
            event.accept()

        def unit(self) -> str:
            return "deg" if self.unit_combo.currentText() == "deg" else "rad"

        def get_one_pose_rad(self) -> list[float]:
            vals = [self.pose_boxes[n].value() for n in ("x", "y", "z", "roll", "pitch", "yaw")]
            if self.unit() == "deg":
                vals[3:] = [math.radians(v) for v in vals[3:]]
            return [float(v) for v in vals]

        def fill_current_pose(self) -> None:
            vals = list(self.last_pose_rad)
            if self.unit() == "deg":
                vals[3:] = [math.degrees(v) for v in vals[3:]]
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), vals):
                self.pose_boxes[name].setValue(float(value))
            self.status_label.setText("已把当前位姿填入 Move One Step。")

        def workpiece_unit(self) -> str:
            return "deg" if self.workpiece_unit_combo.currentText() == "deg" else "rad"

        def fill_current_workpiece_pose(self) -> None:
            if self.last_workpiece_pose_rad is None:
                self.status_label.setText("当前模型中未找到 workpiece_frame，无法填入待磨件位姿输入框。")
                return
            vals = list(self.last_workpiece_pose_rad)
            if self.workpiece_unit() == "deg":
                vals[3:] = [math.degrees(v) for v in vals[3:]]
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), vals):
                self.workpiece_pose_boxes[name].setValue(float(value))
            self.status_label.setText("已把当前待磨件位姿填入输入框。")

        def fill_workpiece_pose_to_move_box(self) -> None:
            if self.last_workpiece_pose_rad is None:
                self.status_label.setText("当前模型中未找到 workpiece_frame，无法填入 Move One Step。")
                return
            vals = list(self.last_workpiece_pose_rad)
            if self.unit() == "deg":
                vals[3:] = [math.degrees(v) for v in vals[3:]]
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), vals):
                self.pose_boxes[name].setValue(float(value))
            self.status_label.setText("已把当前待磨件位姿填入 Move One Step。")

        def get_workpiece_pose_rad(self) -> list[float]:
            vals = [self.workpiece_pose_boxes[n].value() for n in ("x", "y", "z", "roll", "pitch", "yaw")]
            if self.workpiece_unit() == "deg":
                vals[3:] = [math.radians(v) for v in vals[3:]]
            return [float(v) for v in vals]

        def move_workpiece_pose(self) -> None:
            self.post("/workpiece_pose", {"pose": self.get_workpiece_pose_rad()})

        def align_workpiece_handle_step1(self) -> None:
            self.post("/align_workpiece_handle_step1", {})

        def align_workpiece_handle_step2(self) -> None:
            self.post("/align_workpiece_handle_step2", {})

        def execute_ellipsoid_surface_scan(self) -> None:
            self.post("/ellipsoid_surface_scan", {"surface_clearance": float(self.surface_clearance_box.value())})

        def move_one_step(self) -> None:
            self.post("/move", {"mode": "move_one_step", "waypoints": [self.get_one_pose_rad()]})

        def stop_motion(self) -> None:
            self.post("/stop", {})

        def set_gripper(self, value: int) -> None:
            if self.gripper_blocked and value < self.gripper_close_limit:
                self.gripper_slider.blockSignals(True)
                self.gripper_slider.setValue(self.gripper_close_limit)
                self.gripper_slider.blockSignals(False)
                self.status_label.setText(f"gripper 受阻：不能继续闭合，当前限制 = {self.gripper_close_limit}")
                return
            try:
                post_json(f"{self.base_url}/gripper", {"value": int(value)}, timeout=0.25)
            except Exception:
                # Avoid popping message boxes while the slider is being dragged.
                pass

        def open_gripper(self) -> None:
            self.gripper_blocked = False
            self.gripper_close_limit = 0
            self.post("/gripper_open", {})

        def close_gripper(self) -> None:
            self.post("/gripper_close", {})

        def fill_circle_example(self) -> None:
            self.expr_x.setText("0.45 + 0.05*cos(2*pi*t)")
            self.expr_y.setText("0.00 + 0.05*sin(2*pi*t)")
            self.expr_z.setText("0.45 + 0.02*t")
            self.t_start.setValue(0.0)
            self.t_end.setValue(1.0)
            self.samples.setValue(80)

        def generate_function_waypoints(self) -> list[list[float]]:
            n = int(self.samples.value())
            ts = np.linspace(float(self.t_start.value()), float(self.t_end.value()), n)
            if self.function_ori_combo.currentText().startswith("保持当前"):
                rpy = list(self.last_pose_rad[3:])
            else:
                rpy = self.get_one_pose_rad()[3:]
            waypoints: list[list[float]] = []
            for t in ts:
                x = safe_eval_expr(self.expr_x.text(), float(t))
                y = safe_eval_expr(self.expr_y.text(), float(t))
                z = safe_eval_expr(self.expr_z.text(), float(t))
                waypoints.append([x, y, z, float(rpy[0]), float(rpy[1]), float(rpy[2])])
            return waypoints

        def execute_function_path(self) -> None:
            try:
                waypoints = self.generate_function_waypoints()
            except Exception as exc:
                QMessageBox.critical(self, "函数轨迹生成失败", str(exc))
                return
            self.post("/move", {"mode": "function_path", "waypoints": waypoints})

        def poll_state(self) -> None:
            try:
                state = get_json(f"{self.base_url}/state", timeout=0.5)
                self.last_pose_rad = [float(x) for x in state.get("pose_rad", [0] * 6)]
                pose_deg = [float(x) for x in state.get("pose_deg", [0] * 6)]
                self.pose_rad_label.setText(format_pose(self.last_pose_rad))
                self.pose_deg_label.setText(format_pose(pose_deg))
                workpiece_pose_rad = state.get("workpiece_pose_rad")
                workpiece_pose_deg = state.get("workpiece_pose_deg")
                if isinstance(workpiece_pose_rad, list) and len(workpiece_pose_rad) == 6:
                    self.last_workpiece_pose_rad = [float(x) for x in workpiece_pose_rad]
                    self.workpiece_pose_rad_label.setText(format_pose(self.last_workpiece_pose_rad))
                else:
                    self.last_workpiece_pose_rad = None
                    self.workpiece_pose_rad_label.setText("当前模型中未找到 workpiece_frame。")
                if isinstance(workpiece_pose_deg, list) and len(workpiece_pose_deg) == 6:
                    self.last_workpiece_pose_deg = [float(x) for x in workpiece_pose_deg]
                    self.workpiece_pose_deg_label.setText(format_pose(self.last_workpiece_pose_deg))
                else:
                    self.last_workpiece_pose_deg = None
                    self.workpiece_pose_deg_label.setText("当前模型中未找到 workpiece_frame。")
                self.align_step1_error_label.setText(str(state.get("align_step1_error_text", "对齐误差: --")))
                self.workpiece_lock_label.setText(f"待磨件锁定: {state.get('workpiece_lock_status', 'inactive')}")
                self.scan_status_label.setText(f"扫描状态: {state.get('scan_status', 'inactive')}")
                self.err_label.setText(
                    f"pos_err: {float(state.get('pos_err', 0)):.5f} m    "
                    f"ori_err: {float(state.get('ori_err', 0)):.5f} rad"
                )
                wp_i = int(state.get("waypoint_index", 0))
                wp_n = int(state.get("waypoint_count", 0))
                mode = str(state.get("active_mode", "idle"))
                self.status_label.setText(f"{state.get('status', '')}    [{mode} {wp_i}/{wp_n}]")
                g = int(float(state.get("gripper", 255)))
                actual_g = float(state.get("gripper_actual", g))
                command_g = float(state.get("gripper_command", g))
                self.gripper_blocked = bool(state.get("gripper_blocked", False))
                self.gripper_close_limit = max(0, min(255, int(math.ceil(float(state.get("gripper_close_limit", 0))))))
                if self.gripper_blocked and self.gripper_slider.value() < self.gripper_close_limit:
                    self.gripper_slider.blockSignals(True)
                    self.gripper_slider.setValue(self.gripper_close_limit)
                    self.gripper_slider.blockSignals(False)
                elif not self.gripper_slider.isSliderDown() and self.gripper_slider.value() != g:
                    self.gripper_slider.blockSignals(True)
                    self.gripper_slider.setValue(max(0, min(255, g)))
                    self.gripper_slider.blockSignals(False)
                if self.gripper_blocked:
                    self.gripper_state_label.setText(
                        f"actual: {actual_g:.0f}   command: {command_g:.0f}   受阻 limit: {self.gripper_close_limit}"
                    )
                elif bool(state.get("gripper_auto_closing", False)):
                    self.gripper_state_label.setText(f"actual: {actual_g:.0f}   command: {command_g:.0f}   closing")
                else:
                    self.gripper_state_label.setText(f"actual: {actual_g:.0f}   command: {command_g:.0f}")
                show_ee = bool(state.get("show_ee_axes", True))
                show_mocap = bool(state.get("show_mocap", True))
                if self.ee_axes_checkbox.isChecked() != show_ee:
                    self.ee_axes_checkbox.blockSignals(True)
                    self.ee_axes_checkbox.setChecked(show_ee)
                    self.ee_axes_checkbox.blockSignals(False)
                if self.mocap_checkbox.isChecked() != show_mocap:
                    self.mocap_checkbox.blockSignals(True)
                    self.mocap_checkbox.setChecked(show_mocap)
                    self.mocap_checkbox.blockSignals(False)
                show_trail = bool(state.get("show_trail", True))
                if self.trail_checkbox.isChecked() != show_trail:
                    self.trail_checkbox.blockSignals(True)
                    self.trail_checkbox.setChecked(show_trail)
                    self.trail_checkbox.blockSignals(False)
                self.trail_count_label.setText(f"轨迹点: {int(state.get('trail_point_count', 0))}")
            except Exception as exc:
                self.status_label.setText(f"连接控制器失败：{exc}")

    app = QApplication(sys.argv[:1])
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    panel = ControlPanel()
    panel.show()
    panel.raise_()
    panel.activateWindow()
    print("[UI] PySide6 函数轨迹控制面板已打开。", flush=True)
    return int(app.exec())


# ----------------------------- Terminal commands -----------------------------

def print_terminal_help() -> None:
    print(
        "\n[终端命令]\n"
        "  ee_axes on/off      显示/隐藏机械臂末端坐标轴\n"
        "  mocap on/off        显示/隐藏 mocap 和 mocap 坐标轴\n"
        "  vis                 查看显示状态\n"
        "  offset 0.0         设置 mocap z 补偿；默认 0 可让可见 mocap 与末端 site 重合\n"
        "  stop                停止当前轨迹\n"
        "  help                显示命令\n",
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
            with shared.lock:
                if cmd in {"ee_axes on", "ee on", "1 on"}:
                    shared.show_ee_axes = True
                    shared.status = "终端：显示机械臂末端坐标轴"
                    shared.commands.put({"type": "ee_axes", "visible": True})
                    print("[terminal] 已显示机械臂末端坐标轴", flush=True)
                elif cmd in {"ee_axes off", "ee off", "1 off"}:
                    shared.show_ee_axes = False
                    shared.status = "终端：隐藏机械臂末端坐标轴"
                    shared.commands.put({"type": "ee_axes", "visible": False})
                    print("[terminal] 已隐藏机械臂末端坐标轴", flush=True)
                elif cmd in {"mocap on", "target on", "2 on"}:
                    shared.show_mocap = True
                    shared.status = "终端：显示 mocap/目标坐标轴"
                    shared.commands.put({"type": "mocap", "visible": True})
                    print("[terminal] 已显示 mocap/目标坐标轴", flush=True)
                elif cmd in {"mocap off", "target off", "2 off"}:
                    shared.show_mocap = False
                    shared.status = "终端：隐藏 mocap/目标坐标轴"
                    shared.commands.put({"type": "mocap", "visible": False})
                    print("[terminal] 已隐藏 mocap/目标坐标轴", flush=True)
                elif cmd == "vis":
                    print(f"[terminal] ee_axes={'on' if shared.show_ee_axes else 'off'}, mocap={'on' if shared.show_mocap else 'off'}, z_comp={shared.mocap_z_comp:.4f}", flush=True)
                elif cmd.startswith("offset "):
                    try:
                        value = float(cmd.split(maxsplit=1)[1])
                        shared.mocap_z_comp = value
                        shared.status = f"终端：mocap z 补偿 = {value:.4f} m"
                        shared.commands.put({"type": "offset", "mocap_z_comp": value})
                        print(f"[terminal] mocap z 补偿已设为 {value:.4f} m", flush=True)
                    except Exception:
                        print("[terminal] 用法：offset 0.0；如需调试固定偏移可输入 offset 0.0090", flush=True)
                elif cmd == "stop":
                    shared.commands.put({"type": "stop"})
                    print("[terminal] 已请求停止", flush=True)
                elif cmd == "help":
                    print_terminal_help()
                else:
                    print("[terminal] 未识别命令；输入 help 查看命令。", flush=True)

    threading.Thread(target=loop, daemon=True).start()


# ----------------------------- MuJoCo controller -----------------------------

def run_controller(args: argparse.Namespace) -> int:
    import mujoco
    import mujoco.viewer

    cfg = ROBOT_CONFIGS[args.robot]
    xml_path = Path(args.xml or cfg.xml_path)
    print(f"[1/8] 使用 XML: {xml_path}", flush=True)
    if not xml_path.exists():
        raise FileNotFoundError(f"找不到 XML 文件：{xml_path}")

    shared = SharedState(initial_gripper=float(args.gripper_open))
    shared.show_ee_axes = not bool(args.hide_ee_axes)
    shared.show_mocap = not bool(args.hide_mocap)
    shared.mocap_z_comp = float(args.mocap_z_comp)

    server = start_http_server(shared, "127.0.0.1", int(args.port))
    actual_port = int(server.server_address[1])
    print(f"[2/8] 启动本地控制服务：http://127.0.0.1:{actual_port}", flush=True)

    if not args.no_terminal_commands:
        start_terminal_thread(shared)

    ui_proc: Optional[subprocess.Popen[Any]] = None
    if not args.no_ui:
        print("[3/8] 启动 PySide6 UI 子进程，不在 mjpython 进程里创建 QApplication。", flush=True)
        ui_python = find_python_for_ui(args.ui_python)
        ui_proc = launch_ui_subprocess(Path(__file__).resolve(), actual_port, ui_python)
    else:
        print("[3/8] 已禁用 UI 子进程。", flush=True)

    print("[4/8] 加载 MuJoCo 模型...", flush=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    original_geom_rgba = model.geom_rgba.copy()
    original_site_rgba = model.site_rgba.copy()

    model.opt.timestep = dt

    print("[5/8] 绑定 FR3 joints / actuators / site / target...", flush=True)
    site_id = model.site(SITE_NAME).id
    joint_ids, qpos_ids, dof_ids = [], [], []
    for name in cfg.joint_names:
        jid = model.joint(name).id
        joint_ids.append(jid)
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        dof_ids.append(int(model.jnt_dofadr[jid]))
    joint_ids = np.asarray(joint_ids, dtype=int)
    qpos_ids = np.asarray(qpos_ids, dtype=int)
    dof_ids = np.asarray(dof_ids, dtype=int)
    actuator_ids = np.asarray([model.actuator(name).id for name in cfg.actuator_names], dtype=int)

    model.body_gravcomp[:] = 0.0
    if not args.no_gravity_comp and gravity_compensation:
        def body_ancestors(body_id: int) -> list[int]:
            ancestors = [body_id]
            while body_id != 0:
                body_id = int(model.body_parentid[body_id])
                ancestors.append(body_id)
            return ancestors

        def body_depth(body_id: int) -> int:
            return len(body_ancestors(body_id))

        joint_body_ids = [int(model.jnt_bodyid[jid]) for jid in joint_ids]
        common_ancestors = set(body_ancestors(joint_body_ids[0]))
        for body_id in joint_body_ids[1:]:
            common_ancestors &= set(body_ancestors(body_id))
        robot_root_id = max(common_ancestors, key=body_depth) if common_ancestors else 0
        if robot_root_id == 0:
            for body_id in joint_body_ids:
                model.body_gravcomp[body_id] = 1.0
        else:
            for body_id in range(1, model.nbody):
                if robot_root_id in body_ancestors(body_id):
                    model.body_gravcomp[body_id] = 1.0

    gripper_actuator_id: Optional[int] = None
    if cfg.gripper_actuator_name:
        try:
            gripper_actuator_id = model.actuator(cfg.gripper_actuator_name).id
        except KeyError:
            gripper_actuator_id = None
    wheel_spin_actuator_id: Optional[int] = None
    try:
        wheel_spin_actuator_id = model.actuator("wheel_spin").id
    except KeyError:
        wheel_spin_actuator_id = None
    shared.has_gripper = gripper_actuator_id is not None
    finger_qpos_ids: list[int] = []
    for joint_name in FINGER_JOINT_NAMES:
        try:
            jid = model.joint(joint_name).id
            finger_qpos_ids.append(int(model.jnt_qposadr[jid]))
        except KeyError:
            pass

    mocap_id = model.body(MOCAP_NAME).mocapid[0]
    if mocap_id < 0:
        raise ValueError(f"Body '{MOCAP_NAME}' exists but is not a mocap body.")
    workpiece_mocap_id: Optional[int]
    try:
        workpiece_mocap_id = int(model.body(WORKPIECE_MOCAP_NAME).mocapid[0])
        if workpiece_mocap_id < 0:
            workpiece_mocap_id = None
    except KeyError:
        workpiece_mocap_id = None
    key_id = model.key("home").id
    q0_arm = np.array(model.key("home").qpos, copy=True)[qpos_ids]

    ellipsoid_control: Optional[tuple[int, int, int]] = None
    try:
        body_id = int(model.body(ELLIPSOID_BODY_NAME).id)
    except KeyError:
        body_id = -1
    if body_id >= 0:
        jnt_adr = int(model.body_jntadr[body_id])
        jnt_num = int(model.body_jntnum[body_id])
        if jnt_num > 0:
            joint_id = jnt_adr
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                ellipsoid_control = (
                    body_id,
                    int(model.jnt_qposadr[joint_id]),
                    int(model.jnt_dofadr[joint_id]),
                )

    # Visibility ids. Target/mocap visuals are all geoms/sites directly under the target body.
    def maybe_geom(name: str) -> Optional[int]:
        try:
            return int(model.geom(name).id)
        except KeyError:
            return None

    def maybe_site(name: str) -> Optional[int]:
        try:
            return int(model.site(name).id)
        except KeyError:
            return None

    ee_axis_cylinder_geom_id = maybe_geom(EE_AXIS_CYLINDER_GEOM_NAME)
    ee_geom_ids = [
        x
        for x in [
            maybe_geom("ee_axis_x"),
            maybe_geom("ee_axis_y"),
            maybe_geom("ee_axis_z"),
            ee_axis_cylinder_geom_id,
        ]
        if x is not None
    ]
    ee_site_ids = [x for x in [maybe_site(SITE_NAME)] if x is not None]
    workpiece_site_id = maybe_site(WORKPIECE_SITE_NAME)
    ellipsoid_geom_id = maybe_geom(ELLIPSOID_GEOM_NAME)
    wheel_cylinder_geom_id = maybe_geom(WHEEL_CYLINDER_GEOM_NAME)
    target_body_id = int(model.body(MOCAP_NAME).id)
    mocap_geom_ids = [int(i) for i in range(model.ngeom) if int(model.geom_bodyid[i]) == target_body_id]
    mocap_site_ids = [int(i) for i in range(model.nsite) if int(model.site_bodyid[i]) == target_body_id]
    workpiece_target_geom_ids = [
        x
        for x in [
            maybe_geom("workpiece_target_box"),
            maybe_geom("workpiece_target_axis_x"),
            maybe_geom("workpiece_target_axis_y"),
            maybe_geom("workpiece_target_axis_z"),
        ]
        if x is not None
    ]

    def apply_visibility(geom_ids: list[int], site_ids: list[int], visible: bool) -> None:
        for gid in geom_ids:
            model.geom_rgba[gid] = original_geom_rgba[gid]
            if not visible:
                model.geom_rgba[gid, 3] = 0.0
        for sid in site_ids:
            model.site_rgba[sid] = original_site_rgba[sid]
            if not visible:
                model.site_rgba[sid, 3] = 0.0

    def initialize_gripper(value: float) -> None:
        if gripper_actuator_id is None:
            return
        lo, hi = model.actuator_ctrlrange[gripper_actuator_id]
        value = float(np.clip(value, lo, hi))
        data.ctrl[gripper_actuator_id] = value
        finger_open = value / 255.0 * 0.04
        for qpos_id in finger_qpos_ids:
            data.qpos[qpos_id] = finger_open
        mujoco.mj_forward(model, data)

    def actual_gripper_value() -> float:
        if not finger_qpos_ids:
            return float(data.ctrl[gripper_actuator_id]) if gripper_actuator_id is not None else float(args.gripper_open)
        opening = float(np.mean([data.qpos[qpos_id] for qpos_id in finger_qpos_ids]))
        return float(np.clip(opening / 0.04 * 255.0, 0.0, 255.0))

    def current_pose() -> np.ndarray:
        return np.concatenate([np.array(data.site(site_id).xpos, copy=True), rpy_from_mat(data.site(site_id).xmat)])

    def current_workpiece_pose() -> tuple[Optional[list[float]], Optional[list[float]]]:
        if workpiece_site_id is None:
            return None, None
        pose_rad = np.concatenate(
            [
                np.array(data.site(workpiece_site_id).xpos, copy=True),
                rpy_from_mat(data.site(workpiece_site_id).xmat),
            ]
        )
        pose_deg = np.concatenate([pose_rad[:3], np.rad2deg(pose_rad[3:])])
        return [float(x) for x in pose_rad], [float(x) for x in pose_deg]

    def mat_from_quat(quat: np.ndarray) -> np.ndarray:
        mat = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
        return mat.reshape(3, 3)

    def normalize_vector(vec: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(vec, dtype=float)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-9:
            raise ValueError(f"{name} 向量长度过小。")
        return arr / norm

    def pose_command_from_position_mat(position: np.ndarray, mat: np.ndarray) -> PoseCommand:
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).reshape(9))
        raw_pose = np.concatenate([np.asarray(position, dtype=float), rpy_from_mat(mat)])
        return PoseCommand(position=np.asarray(position, dtype=float).copy(), quat=quat.copy(), raw_pose=raw_pose)

    def pose_command_from_position_quat(position: np.ndarray, quat: np.ndarray) -> PoseCommand:
        mat = mat_from_quat(np.asarray(quat, dtype=float))
        raw_pose = np.concatenate([np.asarray(position, dtype=float), rpy_from_mat(mat)])
        return PoseCommand(position=np.asarray(position, dtype=float).copy(), quat=np.asarray(quat, dtype=float).copy(), raw_pose=raw_pose)

    def slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        qa = normalize_vector(np.asarray(q0, dtype=float), "q0")
        qb = normalize_vector(np.asarray(q1, dtype=float), "q1")
        dot = float(np.dot(qa, qb))
        if dot < 0.0:
            qb = -qb
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return normalize_vector((1.0 - alpha) * qa + alpha * qb, "linear quaternion")
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * alpha
        s0 = math.sin(theta_0 - theta) / sin_theta_0
        s1 = math.sin(theta) / sin_theta_0
        return normalize_vector(s0 * qa + s1 * qb, "slerp quaternion")

    def interpolate_commands(start: PoseCommand, end: PoseCommand, steps: int) -> list[PoseCommand]:
        count = max(1, int(steps))
        result: list[PoseCommand] = []
        for i in range(1, count + 1):
            alpha = i / count
            pos = (1.0 - alpha) * start.position + alpha * end.position
            quat = slerp_quat(start.quat, end.quat, alpha)
            result.append(pose_command_from_position_quat(pos, quat))
        return result

    def build_align_step1_goals() -> list[PoseCommand]:
        return [pose_array_to_command(align_step1_fixed_pose_rad)]

    def site_command_from_body_pose(body_pos: np.ndarray, body_mat: np.ndarray, lock_state: WorkpieceLockState) -> PoseCommand:
        if lock_state.mode != "locked" or lock_state.site_to_body_pos is None or lock_state.site_to_body_mat is None:
            raise ValueError("待磨件尚未 locked，无法反算末端目标。")
        site_mat = np.asarray(body_mat, dtype=float) @ lock_state.site_to_body_mat.T
        site_pos = np.asarray(body_pos, dtype=float) - site_mat @ lock_state.site_to_body_pos
        return pose_command_from_position_mat(site_pos - compensation_vec(), site_mat)

    def build_surface_scan_goals(clearance: float, lock_state: WorkpieceLockState) -> tuple[list[PoseCommand], list[str]]:
        if wheel_cylinder_geom_id is None:
            raise ValueError(f"当前模型中没有 {WHEEL_CYLINDER_GEOM_NAME}。")
        if lock_state.mode != "locked":
            raise ValueError("待磨件尚未 locked；请先执行步骤2。")

        wheel_center = np.array(data.geom_xpos[wheel_cylinder_geom_id], copy=True)
        wheel_radius = float(model.geom_size[wheel_cylinder_geom_id][0])
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
        side_dir = np.asarray([1.0, 0.0, 0.0], dtype=float)
        body_pos = wheel_center + side_dir * (wheel_radius + surface_scan_circle_radius + float(clearance))
        body_pos[2] = wheel_center[2]

        commands: list[PoseCommand] = []
        labels: list[str] = []
        for i in range(surface_scan_samples_per_side):
            s = i / max(1, surface_scan_samples_per_side - 1)
            # Centering the body yaw at pi keeps the locked mocap/EE z-axis on the
            # outward side of the +X wheel contact instead of flipping inward.
            yaw = surface_scan_yaw_center - 0.5 * math.pi + math.pi * s
            body_mat = mat_from_quat(quat_from_rpy(0.0, 0.0, yaw))
            commands.append(site_command_from_body_pose(body_pos, body_mat, lock_state))
            labels.append("scanning circle")

        lift = world_up * surface_scan_transition_lift
        final_lift = pose_command_from_position_quat(commands[-1].position + lift, commands[-1].quat)
        commands.append(final_lift)
        labels.append("finished lift")
        return commands, labels

    def body_pose_from_workpiece_site_pose(values: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if workpiece_site_id is None:
            raise ValueError(f"当前模型中没有 {WORKPIECE_SITE_NAME}。")
        if ellipsoid_control is None:
            raise ValueError("当前模型中没有可移动的 ellipsoid。")
        cmd = pose_array_to_command(values)
        site_local_pos = np.array(model.site_pos[workpiece_site_id], copy=True)
        site_local_mat = mat_from_quat(np.array(model.site_quat[workpiece_site_id], copy=True))
        target_site_mat = mat_from_quat(cmd.quat)
        body_mat = target_site_mat @ site_local_mat.T
        body_pos = cmd.position - body_mat @ site_local_pos
        return body_pos, body_mat

    def site_command_from_workpiece_site_pose(values: list[float] | np.ndarray, lock_state: WorkpieceLockState) -> PoseCommand:
        body_pos, body_mat = body_pose_from_workpiece_site_pose(values)
        return site_command_from_body_pose(body_pos, body_mat, lock_state)

    def site_command_from_workpiece_body_pose(values: list[float] | np.ndarray, lock_state: WorkpieceLockState) -> PoseCommand:
        cmd = pose_array_to_command(values)
        body_mat = mat_from_quat(cmd.quat)
        return site_command_from_body_pose(cmd.position, body_mat, lock_state)

    def set_workpiece_pose(values: list[float] | np.ndarray) -> None:
        body_pos, body_mat = body_pose_from_workpiece_site_pose(values)
        apply_workpiece_body_pose(body_pos, body_mat)

    def capture_workpiece_lock(lock_state: WorkpieceLockState) -> None:
        if ellipsoid_control is None:
            raise ValueError("当前模型中没有可移动的 ellipsoid，无法锁定待磨件。")
        body_id, _, _ = ellipsoid_control
        site_pos = np.array(data.site(site_id).xpos, copy=True)
        site_mat = np.asarray(data.site(site_id).xmat, dtype=float).reshape(3, 3)
        body_pos = np.array(data.xpos[body_id], copy=True)
        body_mat = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        lock_state.site_to_body_pos = site_mat.T @ (body_pos - site_pos)
        lock_state.site_to_body_mat = site_mat.T @ body_mat
        lock_state.mode = "locked"

    def apply_workpiece_body_pose(body_pos: np.ndarray, body_mat: np.ndarray) -> None:
        if ellipsoid_control is None:
            return
        _, qpos_adr, qvel_adr = ellipsoid_control
        body_quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(body_quat, np.asarray(body_mat, dtype=float).reshape(9))
        data.qpos[qpos_adr:qpos_adr + 3] = body_pos
        data.qpos[qpos_adr + 3:qpos_adr + 7] = body_quat
        data.qvel[qvel_adr:qvel_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

    def apply_workpiece_lock(lock_state: WorkpieceLockState) -> None:
        if lock_state.mode != "locked" or lock_state.site_to_body_pos is None or lock_state.site_to_body_mat is None:
            return
        if ellipsoid_control is None:
            lock_state.clear()
            return
        site_pos = np.array(data.site(site_id).xpos, copy=True)
        site_mat = np.asarray(data.site(site_id).xmat, dtype=float).reshape(3, 3)
        body_pos = site_pos + site_mat @ lock_state.site_to_body_pos
        body_mat = site_mat @ lock_state.site_to_body_mat
        apply_workpiece_body_pose(body_pos, body_mat)

    def apply_workpiece_mocap_pose() -> None:
        if workpiece_mocap_id is None:
            return
        body_pos = np.array(data.mocap_pos[workpiece_mocap_id], copy=True)
        body_mat = mat_from_quat(np.array(data.mocap_quat[workpiece_mocap_id], copy=True))
        apply_workpiece_body_pose(body_pos, body_mat)

    def orientation_error() -> float:
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        omega = np.zeros(3)
        mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
        mujoco.mju_quat2Vel(omega, error_quat, 1.0)
        return float(np.linalg.norm(omega))

    def compensation_vec() -> np.ndarray:
        # The observed fixed offset is along world Z. Positive value means:
        # controller target = mocap visual origin + [0, 0, z_comp].
        return np.asarray([0.0, 0.0, float(shared.mocap_z_comp)], dtype=float)

    def compensated_target_pos() -> np.ndarray:
        return data.mocap_pos[mocap_id] + compensation_vec()

    def set_mocap_pose(cmd: PoseCommand) -> None:
        # UI/function coordinates are interpreted as visible mocap-frame coordinates.
        # The controller target is shifted by compensation_vec() below.
        data.mocap_pos[mocap_id] = cmd.position
        data.mocap_quat[mocap_id] = cmd.quat

    def set_workpiece_mocap_pose(body_pos: np.ndarray, body_mat: np.ndarray) -> None:
        if workpiece_mocap_id is None:
            return
        body_quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(body_quat, np.asarray(body_mat, dtype=float).reshape(9))
        data.mocap_pos[workpiece_mocap_id] = np.asarray(body_pos, dtype=float)
        data.mocap_quat[workpiece_mocap_id] = body_quat

    def sync_workpiece_mocap_to_body() -> None:
        if workpiece_mocap_id is None or ellipsoid_control is None:
            return
        body_id, _, _ = ellipsoid_control
        body_pos = np.array(data.xpos[body_id], copy=True)
        body_mat = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        set_workpiece_mocap_pose(body_pos, body_mat)

    def command_from_workpiece_mocap(lock_state: WorkpieceLockState) -> PoseCommand:
        if workpiece_mocap_id is None:
            raise ValueError(f"当前模型中没有可拖动的 {WORKPIECE_MOCAP_NAME} mocap。")
        body_pos = np.array(data.mocap_pos[workpiece_mocap_id], copy=True)
        body_mat = mat_from_quat(np.array(data.mocap_quat[workpiece_mocap_id], copy=True))
        return site_command_from_body_pose(body_pos, body_mat, lock_state)

    def snap_mocap_to_site() -> None:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site(site_id).xmat)
        # Keep the visible mocap axes aligned with the visually compensated EE axes.
        data.mocap_pos[mocap_id] = data.site(site_id).xpos - compensation_vec()
        data.mocap_quat[mocap_id] = quat

    def clip_joints(q: np.ndarray) -> None:
        for jid, qpos_id in zip(joint_ids, qpos_ids, strict=True):
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                q[qpos_id] = np.clip(q[qpos_id], lo, hi)

    # mujoco.mj_resetDataKeyframe(model, data, key_id)
    initialize_gripper(float(args.gripper_open))
    sync_workpiece_mocap_to_body()
    snap_mocap_to_site()

    print("[6/8] 打开 MuJoCo viewer...", flush=True)
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=args.show_mujoco_ui,
        show_right_ui=args.show_mujoco_ui,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE if args.show_site_frame_glyphs else mujoco.mjtFrame.mjFRAME_NONE
        try:
            model.vis.scale.framelength = float(args.frame_length)
            model.vis.scale.framewidth = float(args.frame_width)
        except AttributeError:
            pass
        print("[7/8] 控制器已启动。PySide6 控制面板和 MuJoCo viewer 应该都已打开。", flush=True)
        print("[8/8] 函数轨迹功能已启用：在 UI 的 x(t), y(t), z(t) 区域输入函数。", flush=True)

        jac_full = np.zeros((6, model.nv))
        diag = damping * np.eye(6)
        eye_arm = np.eye(len(dof_ids))
        twist = np.zeros(6)
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        motion = MotionState(remaining_goals=[])
        # Streamed function trajectory execution advances the mocap target through
        # sampled points at a fixed interval.
        stream_waypoints: list[PoseCommand] = []
        stream_index = 0
        stream_last_time = 0.0
        stream_step_time = max(0.005, float(args.function_step_time))
        stream_start_hold_time = max(0.0, float(args.stream_start_hold_time))
        stream_started = False
        stream_wait_until = 0.0
        last_state_time = 0.0
        workpiece_lock = WorkpieceLockState()
        surface_scan = SurfaceScanState()
        surface_scan_waypoints: list[PoseCommand] = []
        desired_gripper = float(args.gripper_open)
        auto_gripper_close = False
        gripper_blocked = False
        gripper_close_limit = 0.0
        gripper_stall_since: Optional[float] = None
        last_gripper_actual = actual_gripper_value()
        last_gripper_sample_time = time.time()
        gripper_display_value = desired_gripper
        running = True
        status = "就绪：可拖动 mocap，也可在 UI 中执行函数轨迹。"
        scan_status_text = "inactive"
        trail_strokes: list[list[np.ndarray]] = []
        current_trail_stroke: Optional[list[np.ndarray]] = None
        last_trail_pos: Optional[np.ndarray] = None
        trail_min_distance = 0.003
        trail_max_points = 1500
        trail_radius = 0.002

        def trail_point_count() -> int:
            return sum(len(stroke) for stroke in trail_strokes)

        def trim_trail_points() -> None:
            nonlocal current_trail_stroke
            while trail_point_count() > trail_max_points and trail_strokes:
                excess = trail_point_count() - trail_max_points
                first = trail_strokes[0]
                if len(first) <= excess:
                    removed = trail_strokes.pop(0)
                    if removed is current_trail_stroke:
                        current_trail_stroke = None
                else:
                    del first[:excess]

        def start_trail_stroke() -> None:
            nonlocal current_trail_stroke, last_trail_pos
            current_trail_stroke = []
            trail_strokes.append(current_trail_stroke)
            last_trail_pos = None

        def reset_trail_writer() -> None:
            nonlocal current_trail_stroke, last_trail_pos
            current_trail_stroke = None
            last_trail_pos = None

        def append_trail_point(pos: np.ndarray) -> None:
            nonlocal last_trail_pos
            p = np.asarray(pos, dtype=float).copy()
            if current_trail_stroke is None:
                start_trail_stroke()
            if last_trail_pos is None or float(np.linalg.norm(p - last_trail_pos)) >= trail_min_distance:
                current_trail_stroke.append(p)
                last_trail_pos = p
                trim_trail_points()

        def render_mocap_trail(visible: bool) -> None:
            scn = getattr(viewer, "user_scn", None)
            if scn is None:
                return
            scn.ngeom = 0
            if not visible:
                return
            maxgeom = int(getattr(scn, "maxgeom", 0))
            if maxgeom <= 0:
                return
            segments: list[tuple[np.ndarray, np.ndarray]] = []
            for stroke in trail_strokes:
                if len(stroke) < 2:
                    continue
                segments.extend((stroke[i], stroke[i + 1]) for i in range(len(stroke) - 1))
            if not segments:
                return
            segment_count = min(len(segments), maxgeom)
            visible_segments = segments[-segment_count:]
            mat = np.eye(3, dtype=float).reshape(9)
            size = np.zeros(3, dtype=float)
            pos = np.zeros(3, dtype=float)
            for segment_i, (start, end) in enumerate(visible_segments, start=1):
                if scn.ngeom >= maxgeom:
                    break
                alpha = 0.25 + 0.55 * (segment_i / max(1, segment_count))
                rgba = np.asarray([0.0, 0.85, 1.0, alpha], dtype=np.float32)
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, size, pos, mat, rgba)
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    trail_radius,
                    start,
                    end,
                )
                scn.ngeom += 1

        def reset_motion_execution() -> None:
            nonlocal stream_waypoints, stream_index, stream_last_time, stream_started, stream_wait_until, surface_scan_waypoints
            stream_waypoints = []
            stream_index = 0
            stream_last_time = 0.0
            stream_started = False
            stream_wait_until = 0.0
            surface_scan.clear()
            surface_scan_waypoints = []
            motion.clear()

        while viewer.is_running() and running:
            step_start = time.time()

            while True:
                try:
                    command = shared.commands.get_nowait()
                except queue.Empty:
                    break
                try:
                    ctype = command.get("type")
                    if ctype == "move":
                        surface_scan.clear()
                        surface_scan_waypoints = []
                        scan_status_text = "inactive"
                        waypoints = parse_waypoints_rad(command.get("waypoints", []))
                        if waypoints:
                            mode = str(command.get("mode", "move"))
                            motion.active_mode = mode
                            motion.current_goal = waypoints[0]
                            motion.active_goal_index = 1
                            motion.total_goal_count = len(waypoints)
                            set_mocap_pose(motion.current_goal)

                            if mode == "function_path":
                                # Function path is streamed in time instead of waiting for each
                                # sampled waypoint to be exactly reached.
                                stream_waypoints = waypoints
                                stream_index = 0
                                stream_started = False
                                stream_wait_until = time.time() + stream_start_hold_time
                                stream_last_time = 0.0
                                motion.current_goal = None
                                motion.active_goal_index = 0
                                motion.remaining_goals = []
                                status = (
                                    f"准备执行函数轨迹：mocap 先保持当前位置 {stream_start_hold_time:.2f}s，"
                                    f"机械臂先移动到当前 mocap 初始点；随后每 {stream_step_time:.3f}s 推进一个点"
                                )
                            else:
                                # move_one_step still waits until the target is reached.
                                stream_waypoints = []
                                stream_index = 0
                                motion.remaining_goals = waypoints[1:]
                                status = f"开始执行 {motion.active_mode}: waypoint {motion.active_goal_index}/{motion.total_goal_count}"
                    elif ctype == "align_workpiece_handle_step1":
                        goals = build_align_step1_goals()
                        if goals:
                            reset_motion_execution()
                            motion.active_mode = "align_step1_fixed"
                            motion.current_goal = goals[0]
                            motion.remaining_goals = goals[1:]
                            motion.active_goal_index = 1
                            motion.total_goal_count = len(goals)
                            set_mocap_pose(motion.current_goal)
                            status = f"开始执行对齐步骤1：移动到固定目标 {align_step1_fixed_pose_deg_text}"
                    elif ctype == "ellipsoid_surface_scan":
                        clearance = float(np.clip(float(command.get("surface_clearance", 0.0)), -0.020, 0.020))
                        if workpiece_lock.mode != "locked":
                            scan_status_text = "inactive"
                            status = "圆周扫描失败：待磨件尚未 locked；请先执行步骤2。"
                        else:
                            goals, labels = build_surface_scan_goals(clearance, workpiece_lock)
                            reset_motion_execution()
                            surface_scan_waypoints = goals
                            surface_scan.active = True
                            surface_scan.streaming = False
                            surface_scan.index = 0
                            surface_scan.clearance = clearance
                            surface_scan.labels = labels
                            motion.active_mode = "ellipsoid_surface_scan"
                            motion.current_goal = goals[0]
                            motion.remaining_goals = []
                            motion.active_goal_index = 1
                            motion.total_goal_count = len(goals)
                            set_mocap_pose(motion.current_goal)
                            scan_status_text = (
                                f"waiting_start, point 1/{len(goals)}, clearance={clearance:.4f}m"
                            )
                            status = f"圆周扫描准备中：等待第一个点到达，surface_clearance={clearance:.4f}m"
                    elif ctype == "align_workpiece_handle_step2":
                        if ellipsoid_control is None:
                            workpiece_lock.clear()
                            status = "对齐步骤2失败：当前模型中没有可移动的待磨件 freejoint。"
                        else:
                            surface_scan.clear()
                            surface_scan_waypoints = []
                            scan_status_text = "inactive"
                            if gripper_actuator_id is not None:
                                lo, _ = model.actuator_ctrlrange[gripper_actuator_id]
                                desired_gripper = float(lo)
                            else:
                                desired_gripper = 0.0
                            auto_gripper_close = True
                            gripper_blocked = False
                            gripper_close_limit = 0.0
                            gripper_stall_since = None
                            workpiece_lock.start_closing()
                            status = "对齐步骤2：正在关闭 gripper，等待闭合或受阻后锁定待磨件。"
                    elif ctype == "stop":
                        stream_waypoints = []
                        stream_index = 0
                        stream_started = False
                        stream_wait_until = 0.0
                        reset_trail_writer()
                        motion.clear()
                        surface_scan.clear()
                        surface_scan_waypoints = []
                        scan_status_text = "inactive"
                        snap_mocap_to_site()
                        status = "已停止当前任务，并把 target 同步到当前末端位姿。"
                    elif ctype == "gripper":
                        desired_gripper = float(command.get("value", desired_gripper))
                        auto_gripper_close = False
                    elif ctype == "gripper_open":
                        if gripper_actuator_id is not None:
                            _, hi = model.actuator_ctrlrange[gripper_actuator_id]
                            desired_gripper = float(hi)
                        else:
                            desired_gripper = float(args.gripper_open)
                        auto_gripper_close = False
                        gripper_blocked = False
                        gripper_close_limit = 0.0
                        gripper_stall_since = None
                        workpiece_lock.clear()
                        sync_workpiece_mocap_to_body()
                        surface_scan.clear()
                        surface_scan_waypoints = []
                        scan_status_text = "inactive"
                        status = "gripper open：完全打开"
                    elif ctype == "gripper_close":
                        if gripper_actuator_id is not None:
                            lo, _ = model.actuator_ctrlrange[gripper_actuator_id]
                            desired_gripper = float(lo)
                        else:
                            desired_gripper = 0.0
                        auto_gripper_close = True
                        gripper_blocked = False
                        gripper_close_limit = 0.0
                        gripper_stall_since = None
                        status = "gripper close：正在关闭直到受阻或完全闭合"
                    elif ctype == "ee_axes":
                        shared.show_ee_axes = bool(command.get("visible", True))
                    elif ctype == "mocap":
                        shared.show_mocap = bool(command.get("visible", True))
                    elif ctype == "trail":
                        shared.show_trail = bool(command.get("enabled", True))
                        reset_trail_writer()
                        status = "开始记录 mocap 中心轨迹" if shared.show_trail else "停止记录 mocap 中心轨迹"
                    elif ctype == "clear_trail":
                        trail_strokes.clear()
                        reset_trail_writer()
                        status = "已清除 mocap 中心轨迹。"
                    elif ctype == "offset":
                        shared.mocap_z_comp = float(command.get("mocap_z_comp", shared.mocap_z_comp))
                        status = f"mocap z 补偿 = {shared.mocap_z_comp:.4f} m"
                    elif ctype == "workpiece_pose":
                        pose_values = command.get("pose", [])
                        if workpiece_lock.mode == "closing":
                            status = "待磨件正在锁定；请等待步骤2完成后再移动待磨件。"
                        elif workpiece_lock.mode == "locked":
                            cmd = pose_array_to_command(pose_values)
                            set_workpiece_mocap_pose(cmd.position, mat_from_quat(cmd.quat))
                            goal = command_from_workpiece_mocap(workpiece_lock)
                            reset_motion_execution()
                            scan_status_text = "inactive"
                            motion.active_mode = "locked_workpiece_pose"
                            motion.current_goal = goal
                            motion.remaining_goals = []
                            motion.active_goal_index = 1
                            motion.total_goal_count = 1
                            set_mocap_pose(goal)
                            status = f"locked：已按双球球心目标反算机械臂末端，开始移动到 {format_pose(pose_values)}"
                        else:
                            set_workpiece_pose(pose_values)
                            status = f"已移动待磨件: {format_pose(pose_values)}"
                    elif ctype == "quit":
                        running = False
                except Exception as exc:
                    status = f"命令执行失败：{exc}"

            with shared.lock:
                show_ee = bool(shared.show_ee_axes)
                show_mocap = bool(shared.show_mocap)
                show_trail = bool(shared.show_trail)
            apply_visibility(ee_geom_ids, ee_site_ids, show_ee)
            apply_visibility(mocap_geom_ids, mocap_site_ids, show_mocap)
            apply_visibility(workpiece_target_geom_ids, [], show_mocap and workpiece_lock.mode == "locked")

            if workpiece_lock.mode == "locked" and workpiece_mocap_id is not None and motion.active_mode in {"", "locked_workpiece_pose", "locked_workpiece_mocap_drag"}:
                try:
                    goal = command_from_workpiece_mocap(workpiece_lock)
                    motion.active_mode = "locked_workpiece_mocap_drag"
                    motion.current_goal = goal
                    motion.remaining_goals = []
                    motion.active_goal_index = 1
                    motion.total_goal_count = 1
                    set_mocap_pose(goal)
                except Exception as exc:
                    status = f"读取球心 mocap target 失败：{exc}"

            if gripper_actuator_id is not None:
                lo, hi = model.actuator_ctrlrange[gripper_actuator_id]
                data.ctrl[gripper_actuator_id] = float(np.clip(desired_gripper, lo, hi))
            if wheel_spin_actuator_id is not None:
                lo, hi = model.actuator_ctrlrange[wheel_spin_actuator_id]
                data.ctrl[wheel_spin_actuator_id] = float(np.clip(2.0, lo, hi))

            # For streamed function_path: first hold the current mocap pose for
            # stream_start_hold_time, then send the first sampled waypoint.
            if stream_waypoints and not stream_started:
                now_for_stream = time.time()
                if now_for_stream >= stream_wait_until:
                    stream_started = True
                    stream_index = 0
                    motion.current_goal = stream_waypoints[0]
                    motion.active_goal_index = 1
                    set_mocap_pose(motion.current_goal)
                    stream_last_time = now_for_stream
                    status = f"开始执行函数轨迹：采样点 1/{motion.total_goal_count}"
                else:
                    remain = max(0.0, stream_wait_until - now_for_stream)
                    status = f"函数轨迹准备中：mocap 保持当前初始点，还剩 {remain:.2f}s 后开始轨迹"

            if motion.current_goal is not None:
                pos_err_goal = float(np.linalg.norm(compensated_target_pos() - data.site(site_id).xpos))
                ori_err_goal = orientation_error()

                if motion.active_mode == "ellipsoid_surface_scan" and surface_scan.active and surface_scan_waypoints:
                    now_for_scan = time.time()
                    if not surface_scan.streaming:
                        if pos_err_goal < args.position_tolerance and ori_err_goal < args.orientation_tolerance:
                            surface_scan.streaming = True
                            surface_scan.last_step_at = now_for_scan
                            scan_status_text = (
                                f"{surface_scan.label()}, point {surface_scan.index + 1}/{len(surface_scan_waypoints)}, "
                                f"clearance={surface_scan.clearance:.4f}m"
                            )
                            status = f"圆周扫描开始：{scan_status_text}"
                        else:
                            scan_status_text = (
                                f"waiting_start, point 1/{len(surface_scan_waypoints)}, "
                                f"clearance={surface_scan.clearance:.4f}m"
                            )
                            status = f"圆周扫描准备中：等待第一个点到达。{scan_status_text}"
                    elif surface_scan.index < len(surface_scan_waypoints) - 1:
                        if now_for_scan - surface_scan.last_step_at >= surface_scan_step_time:
                            surface_scan.index += 1
                            motion.current_goal = surface_scan_waypoints[surface_scan.index]
                            motion.active_goal_index = surface_scan.index + 1
                            set_mocap_pose(motion.current_goal)
                            surface_scan.last_step_at = now_for_scan
                            scan_status_text = (
                                f"{surface_scan.label()}, point {surface_scan.index + 1}/{len(surface_scan_waypoints)}, "
                                f"clearance={surface_scan.clearance:.4f}m"
                            )
                            status = f"圆周扫描执行中：{scan_status_text}"
                    else:
                        if pos_err_goal < args.position_tolerance and ori_err_goal < args.orientation_tolerance:
                            scan_status_text = (
                                f"finished, point {surface_scan.index + 1}/{len(surface_scan_waypoints)}, "
                                f"clearance={surface_scan.clearance:.4f}m"
                            )
                            status = f"圆周扫描完成：{scan_status_text}"
                            surface_scan.clear()
                            surface_scan_waypoints = []
                            motion.clear()
                elif motion.active_mode == "function_path" and stream_waypoints and stream_started:
                    now_for_stream = time.time()
                    if stream_index < len(stream_waypoints) - 1 and now_for_stream - stream_last_time >= stream_step_time:
                        stream_index += 1
                        motion.current_goal = stream_waypoints[stream_index]
                        motion.active_goal_index = stream_index + 1
                        set_mocap_pose(motion.current_goal)
                        stream_last_time = now_for_stream
                        status = f"函数轨迹执行中：采样点 {motion.active_goal_index}/{motion.total_goal_count}"
                    elif stream_index >= len(stream_waypoints) - 1:
                        # After the final sampled point has been sent, wait until the
                        # robot actually catches up, then mark the path as complete.
                        if pos_err_goal < args.position_tolerance and ori_err_goal < args.orientation_tolerance:
                            status = "函数轨迹执行完成。"
                            stream_waypoints = []
                            stream_index = 0
                            stream_started = False
                            stream_wait_until = 0.0
                            motion.clear()
                else:
                    reached_goal = pos_err_goal < args.position_tolerance and ori_err_goal < args.orientation_tolerance
                    if reached_goal:
                        if motion.remaining_goals:
                            motion.current_goal = motion.remaining_goals.pop(0)
                            motion.active_goal_index += 1
                            set_mocap_pose(motion.current_goal)
                            status = f"继续执行 waypoint {motion.active_goal_index}/{motion.total_goal_count}"
                        else:
                            if motion.active_mode == "align_step1_fixed":
                                status = f"对齐步骤1执行完成：已移动到固定目标 {align_step1_fixed_pose_deg_text}"
                            elif motion.active_mode == "locked_workpiece_pose":
                                status = "locked 待磨件目标执行完成：机械臂已移动到反算末端目标。"
                            else:
                                status = f"{motion.active_mode} 执行完成。"
                            motion.clear()

            dx = compensated_target_pos() - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            mujoco.mj_jacSite(model, data, jac_full[:3], jac_full[3:], site_id)
            jac = jac_full[:, dof_ids]
            try:
                jac_dls = jac.T @ np.linalg.solve(jac @ jac.T + diag, np.eye(6))
                dq_arm = jac_dls @ twist
                q_arm = data.qpos[qpos_ids]
                nullspace_projector = eye_arm - jac_dls @ jac
                dq_arm += nullspace_projector @ (Kn * (q0_arm - q_arm))
            except np.linalg.LinAlgError:
                dq_arm = np.zeros(len(dof_ids))
                status = "警告：Jacobian 求解失败，已跳过当前控制步。"

            dq_abs_max = np.abs(dq_arm).max()
            if dq_abs_max > max_angvel:
                dq_arm *= max_angvel / dq_abs_max

            q = data.qpos.copy()
            dq_full = np.zeros(model.nv)
            dq_full[dof_ids] = dq_arm
            mujoco.mj_integratePos(model, q, dq_full, integration_dt)
            clip_joints(q)
            data.ctrl[actuator_ids] = q[qpos_ids]
            mujoco.mj_step(model, data)

            actual_gripper = actual_gripper_value()
            now_for_gripper = time.time()
            gripper_dt = max(1e-6, now_for_gripper - last_gripper_sample_time)
            gripper_speed = abs(actual_gripper - last_gripper_actual) / gripper_dt
            closing_requested = desired_gripper < actual_gripper - 3.0
            opening_requested = desired_gripper > actual_gripper + 3.0
            fully_closed = actual_gripper <= 2.0

            if opening_requested:
                gripper_blocked = False
                gripper_close_limit = 0.0
                gripper_stall_since = None
                auto_gripper_close = False
            elif closing_requested and not fully_closed:
                if gripper_speed < 2.0:
                    if gripper_stall_since is None:
                        gripper_stall_since = now_for_gripper
                    elif now_for_gripper - gripper_stall_since >= 0.20:
                        gripper_blocked = True
                        gripper_close_limit = float(np.clip(actual_gripper, 0.0, 255.0))
                        if auto_gripper_close:
                            auto_gripper_close = False
                        status = f"gripper 受阻：实际开度 {actual_gripper:.0f}，已阻止继续闭合滑条"
                else:
                    gripper_stall_since = None
            else:
                gripper_stall_since = None
                if fully_closed:
                    was_auto_closing = auto_gripper_close
                    gripper_blocked = False
                    gripper_close_limit = 0.0
                    auto_gripper_close = False
                    if was_auto_closing and desired_gripper <= 2.0:
                        status = "gripper 已完全闭合。"

            if workpiece_lock.mode == "closing" and (fully_closed or gripper_blocked):
                try:
                    capture_workpiece_lock(workpiece_lock)
                    sync_workpiece_mocap_to_body()
                    status = "对齐步骤2完成：gripper 已闭合，待磨件已虚拟刚性锁定。"
                except Exception as exc:
                    workpiece_lock.clear()
                    status = f"对齐步骤2失败：{exc}"

            if gripper_blocked:
                gripper_close_limit = float(np.clip(actual_gripper, 0.0, 255.0))
            if gripper_blocked:
                gripper_display_value = gripper_close_limit
            elif auto_gripper_close:
                gripper_display_value = actual_gripper
            else:
                gripper_display_value = desired_gripper
            last_gripper_actual = actual_gripper
            last_gripper_sample_time = now_for_gripper
            if workpiece_lock.mode == "locked" and motion.active_mode == "locked_workpiece_mocap_drag":
                apply_workpiece_mocap_pose()
            else:
                apply_workpiece_lock(workpiece_lock)

            if show_trail:
                append_trail_point(data.site(site_id).xpos - compensation_vec())
            else:
                last_trail_pos = None
            render_mocap_trail(show_trail)
            viewer.sync()

            now = time.time()
            if now - last_state_time > 0.05:
                pose_rad = current_pose()
                pose_deg = np.concatenate([pose_rad[:3], np.rad2deg(pose_rad[3:])])
                workpiece_pose_rad, workpiece_pose_deg = current_workpiece_pose()
                align_error_text = f"步骤1固定目标: {align_step1_fixed_pose_deg_text}"
                status_for_shared = status
                shared.update(
                    pose_rad=[float(x) for x in pose_rad],
                    pose_deg=[float(x) for x in pose_deg],
                    pos_err=float(np.linalg.norm(compensated_target_pos() - data.site(site_id).xpos)),
                    ori_err=orientation_error(),
                    status=status_for_shared,
                    gripper_value=float(gripper_display_value),
                    gripper_command_value=float(desired_gripper),
                    gripper_actual_value=float(actual_gripper),
                    gripper_blocked=bool(gripper_blocked),
                    gripper_close_limit=float(gripper_close_limit),
                    gripper_auto_closing=bool(auto_gripper_close),
                    mocap_z_comp=float(shared.mocap_z_comp),
                    viewer_running=True,
                    waypoint_index=int(motion.active_goal_index),
                    waypoint_count=int(motion.total_goal_count),
                    active_mode=motion.active_mode or "idle",
                    show_trail=show_trail,
                    trail_point_count=trail_point_count(),
                    workpiece_pose_rad=workpiece_pose_rad,
                    workpiece_pose_deg=workpiece_pose_deg,
                    align_step1_error_text=align_error_text,
                    workpiece_lock_status=workpiece_lock.mode,
                    scan_status=scan_status_text,
                )
                last_state_time = now

            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    shared.update(status="MuJoCo viewer 已关闭。", viewer_running=False)
    try:
        server.shutdown()
    except Exception:
        pass
    if ui_proc is not None and ui_proc.poll() is None:
        try:
            ui_proc.terminate()
        except Exception:
            pass
    print("已退出。", flush=True)
    return 0


# ----------------------------- CLI -------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FR3 differential IK + PySide6 UI + xyz-function trajectory")
    parser.add_argument("--ui-client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ui-python", type=str, default=None, help="Python interpreter used for the PySide6 UI subprocess")
    parser.add_argument("--no-ui", action="store_true", help="Do not launch the PySide6 UI subprocess")
    parser.add_argument("--robot", choices=sorted(ROBOT_CONFIGS), default="fr3")
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--no-gravity-comp", action="store_true")
    parser.add_argument("--show-mujoco-ui", action="store_true")
    parser.add_argument("--gripper-open", type=float, default=255.0)
    parser.add_argument("--position-tolerance", type=float, default=0.006)
    parser.add_argument("--orientation-tolerance", type=float, default=0.045)
    parser.add_argument("--frame-length", type=float, default=0.09)
    parser.add_argument("--frame-width", type=float, default=0.003)
    parser.add_argument("--show-site-frame-glyphs", action="store_true")
    parser.add_argument("--hide-ee-axes", action="store_true")
    parser.add_argument("--hide-mocap", action="store_true")
    parser.add_argument("--mocap-z-comp", type=float, default=0.0, help="Fixed Z compensation between visible mocap axes and controlled EE site. Default 0 keeps the visible mocap origin aligned with the controlled EE site.")
    parser.add_argument("--function-step-time", type=float, default=0.06, help="Time interval between sampled points during direct function-path execution. Smaller is faster. Default: 0.06s")
    parser.add_argument("--stream-start-hold-time", type=float, default=2.0, help="Before streaming function_path, keep mocap at its current pose for this many seconds. Default: 2.0s")
    parser.add_argument("--no-terminal-commands", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ui_client:
        return run_ui_client(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
