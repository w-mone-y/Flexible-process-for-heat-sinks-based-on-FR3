"""FR3 differential IK controller with PySide6 UI and xyz-function trajectory.

Run from your project root with mjpython, for example:

    mjpython diffik_nullspace_fr3_pyside6_ui_curve_offset_fixed.py \
        --xml franka_fr3/scene_fr3_with_gripper_grasp_objects_fixed.xml

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
  - cube/cylinder world pose display and direct 6D pose reset
  - move one 6D pose
  - move multiple 6D waypoints
  - xyz-function trajectory: x(t), y(t), z(t), t_start, t_end, samples
  - direct function trajectory is time-streamed so the target visibly moves along the curve
  - Multi Step waypoints are also time-streamed so they do not get stuck at the first waypoint
  - function_path and Multi Step first hold the current mocap pose for 1s before streaming
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

# Names expected in your FR3+gripper XML.
SITE_NAME = "attachment_site"
MOCAP_NAME = "target"
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
GRASP_OBJECT_BODY_NAMES = {
    "cube": "grasp_cube",
    "cylinder": "grasp_cylinder",
}


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
class RobotConfig:
    xml_path: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    gripper_actuator_name: Optional[str] = "gripper"


ROBOT_CONFIGS = {
    "fr3": RobotConfig(
        xml_path="franka_fr3/scene_fr3_with_gripper_grasp_objects_fixed.xml",
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
        self.show_trail = True
        self.mocap_z_comp = 0.0090
        self.has_gripper = True
        self.viewer_running = False
        self.waypoint_index = 0
        self.waypoint_count = 0
        self.active_mode = "idle"
        self.trail_point_count = 0
        self.object_names: list[str] = []
        self.object_poses_rad: dict[str, list[float]] = {}
        self.object_poses_deg: dict[str, list[float]] = {}
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
                "object_names": list(self.object_names),
                "object_poses_rad": {name: list(pose) for name, pose in self.object_poses_rad.items()},
                "object_poses_deg": {name: list(pose) for name, pose in self.object_poses_deg.items()},
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
                value = float(payload.get("mocap_z_comp", 0.0090))
                self.shared.update(mocap_z_comp=value, status=f"mocap z 补偿 = {value:.4f} m")
                self.shared.commands.put({"type": "offset", "mocap_z_comp": value})
                self._send_json({"ok": True, "mocap_z_comp": value})
            elif self.path == "/object_pose":
                object_name = str(payload.get("object", "")).strip()
                pose = payload.get("pose", [])
                if object_name not in GRASP_OBJECT_BODY_NAMES:
                    raise ValueError("object must be one of: " + ", ".join(GRASP_OBJECT_BODY_NAMES))
                if not isinstance(pose, list) or len(pose) != 6:
                    raise ValueError("pose must have 6 elements: x y z roll pitch yaw")
                values = [float(x) for x in pose]
                self.shared.commands.put({"type": "object_pose", "object": object_name, "pose": values})
                self.shared.update(status=f"请求移动 {object_name}: {format_pose(values)}")
                self._send_json({"ok": True, "object": object_name, "pose": values})
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
            QTextEdit,
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
            self.last_object_poses_rad: dict[str, list[float]] = {}
            self.last_object_poses_deg: dict[str, list[float]] = {}
            self.gripper_blocked = False
            self.gripper_close_limit = 0
            self.default_unit = "rad"
            self._build()

        def _build(self) -> None:
            self.setWindowTitle("FR3 PySide6 控制面板 - 函数轨迹")
            self.resize(820, 980)
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

            title = QLabel("FR3 末端位姿 / 物体位姿 / 函数轨迹 / Gripper 控制")
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

            # Grasp object pose.
            object_group = QGroupBox("抓取物体世界坐标六维位姿")
            object_layout = QVBoxLayout(object_group)
            object_row = QHBoxLayout()
            object_row.addWidget(QLabel("物体："))
            self.object_combo = QComboBox()
            self.object_combo.addItem("cube", "cube")
            self.object_combo.addItem("cylinder", "cylinder")
            self.object_combo.currentIndexChanged.connect(self.update_object_pose_display)
            object_row.addWidget(self.object_combo)
            object_row.addWidget(QLabel("输入姿态角单位："))
            self.object_unit_combo = QComboBox()
            self.object_unit_combo.addItems(["rad", "deg"])
            object_row.addWidget(self.object_unit_combo)
            object_row.addStretch(1)
            object_layout.addLayout(object_row)
            object_layout.addWidget(QLabel("rad:  x y z roll pitch yaw"))
            self.object_pose_rad_label = QLabel("等待仿真数据...")
            self.object_pose_rad_label.setObjectName("MonoLabel")
            object_layout.addWidget(self.object_pose_rad_label)
            object_layout.addWidget(QLabel("deg:  x y z roll pitch yaw"))
            self.object_pose_deg_label = QLabel("等待仿真数据...")
            self.object_pose_deg_label.setObjectName("MonoLabel")
            object_layout.addWidget(self.object_pose_deg_label)

            self.object_pose_boxes: dict[str, QDoubleSpinBox] = {}
            object_grid = QGridLayout()
            object_defaults = {"x": 0.62, "y": -0.075, "z": 0.220, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            for col, name in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
                box = QDoubleSpinBox()
                box.setDecimals(6)
                box.setRange(-10000, 10000)
                box.setSingleStep(0.01 if name in {"x", "y", "z"} else 0.05)
                box.setValue(object_defaults[name])
                self.object_pose_boxes[name] = box
                object_grid.addWidget(QLabel(name), 0, col)
                object_grid.addWidget(box, 1, col)
            object_layout.addLayout(object_grid)

            object_btns = QHBoxLayout()
            fill_object_btn = QPushButton("把当前物体位姿填入")
            fill_object_btn.clicked.connect(self.fill_current_object_pose)
            move_object_btn = QPushButton("移动物体到该位姿")
            move_object_btn.clicked.connect(self.move_object_pose)
            object_btns.addWidget(fill_object_btn)
            object_btns.addWidget(move_object_btn)
            object_btns.addStretch(1)
            object_layout.addLayout(object_btns)
            layout.addWidget(object_group)

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
            preview_btn = QPushButton("生成到 Multi Step 文本框")
            preview_btn.clicked.connect(self.preview_function_path)
            exec_btn = QPushButton("直接执行函数轨迹")
            exec_btn.clicked.connect(self.execute_function_path)
            sample_btn = QPushButton("填入圆形示例")
            sample_btn.clicked.connect(self.fill_circle_example)
            func_btns.addWidget(preview_btn)
            func_btns.addWidget(exec_btn)
            func_btns.addWidget(sample_btn)
            func_btns.addStretch(1)
            func_layout.addLayout(func_btns)
            layout.addWidget(func_group)

            # Multi step.
            multi_group = QGroupBox("Move Multi Step")
            multi_layout = QVBoxLayout(multi_group)
            self.multi_text = QTextEdit()
            self.multi_text.setAcceptRichText(False)
            self.multi_text.setPlainText("0.45 0.00 0.45 3.1416 0 0\n0.45 0.10 0.45 3.1416 0 0")
            multi_layout.addWidget(self.multi_text, stretch=1)
            multi_row = QHBoxLayout()
            mbtn = QPushButton("按顺序执行 waypoints")
            mbtn.clicked.connect(self.move_multi_step)
            multi_row.addWidget(mbtn)
            cbtn = QPushButton("清空")
            cbtn.clicked.connect(self.multi_text.clear)
            multi_row.addWidget(cbtn)
            multi_row.addStretch(1)
            multi_layout.addLayout(multi_row)
            layout.addWidget(multi_group, stretch=1)

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
            layout.addWidget(vis_group)

            trail_group = QGroupBox("Mocap 中心运动轨迹")
            trail_layout = QHBoxLayout(trail_group)
            self.trail_checkbox = QCheckBox("留下 mocap 中心轨迹")
            self.trail_checkbox.setChecked(True)
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

        def selected_object_key(self) -> str:
            data = self.object_combo.currentData()
            return str(data if data is not None else self.object_combo.currentText()).strip()

        def object_unit(self) -> str:
            return "deg" if self.object_unit_combo.currentText() == "deg" else "rad"

        def update_object_pose_display(self, *_args: Any) -> None:
            key = self.selected_object_key()
            pose_rad = self.last_object_poses_rad.get(key)
            pose_deg = self.last_object_poses_deg.get(key)
            if pose_rad is None or pose_deg is None:
                self.object_pose_rad_label.setText("该物体未在当前模型中找到。")
                self.object_pose_deg_label.setText("该物体未在当前模型中找到。")
                return
            self.object_pose_rad_label.setText(format_pose(pose_rad))
            self.object_pose_deg_label.setText(format_pose(pose_deg))

        def fill_current_object_pose(self) -> None:
            key = self.selected_object_key()
            vals = list(self.last_object_poses_rad.get(key, [0.0] * 6))
            if self.object_unit() == "deg":
                vals[3:] = [math.degrees(v) for v in vals[3:]]
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), vals):
                self.object_pose_boxes[name].setValue(float(value))
            self.status_label.setText(f"已把当前 {key} 位姿填入物体位姿输入框。")

        def get_object_pose_rad(self) -> list[float]:
            vals = [self.object_pose_boxes[n].value() for n in ("x", "y", "z", "roll", "pitch", "yaw")]
            if self.object_unit() == "deg":
                vals[3:] = [math.radians(v) for v in vals[3:]]
            return [float(v) for v in vals]

        def move_object_pose(self) -> None:
            self.post("/object_pose", {"object": self.selected_object_key(), "pose": self.get_object_pose_rad()})

        def move_one_step(self) -> None:
            self.post("/move", {"mode": "move_one_step", "waypoints": [self.get_one_pose_rad()]})

        def parse_multi_line(self, line: str) -> list[float]:
            tokens = line.replace(",", " ").strip().split()
            if not tokens:
                raise ValueError("空行")
            unit = self.unit()
            if tokens[0].lower() in {"rad", "deg"}:
                unit = tokens[0].lower()
                tokens = tokens[1:]
            if len(tokens) != 6:
                raise ValueError("每行需要 6 个数字：x y z roll pitch yaw")
            vals = [float(x) for x in tokens]
            if unit == "deg":
                vals[3:] = [math.radians(v) for v in vals[3:]]
            return vals

        def move_multi_step(self) -> None:
            waypoints: list[list[float]] = []
            errors: list[str] = []
            for i, line in enumerate(self.multi_text.toPlainText().splitlines(), start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    waypoints.append(self.parse_multi_line(line))
                except Exception as exc:
                    errors.append(f"第 {i} 行：{exc}")
            if errors:
                QMessageBox.critical(self, "waypoint 输入无效", "\n".join(errors[:8]))
                return
            if not waypoints:
                QMessageBox.warning(self, "没有 waypoint", "请至少输入一行 waypoint。")
                return
            self.post("/move", {"mode": "move_multi_step", "waypoints": waypoints})

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

        def preview_function_path(self) -> None:
            try:
                waypoints = self.generate_function_waypoints()
            except Exception as exc:
                QMessageBox.critical(self, "函数轨迹生成失败", str(exc))
                return
            self.multi_text.setPlainText("\n".join(format_pose(wp) for wp in waypoints))
            self.status_label.setText(f"已生成 {len(waypoints)} 个函数轨迹 waypoint 到 Multi Step 文本框。")

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
                object_poses_rad = state.get("object_poses_rad", {})
                object_poses_deg = state.get("object_poses_deg", {})
                if isinstance(object_poses_rad, dict):
                    self.last_object_poses_rad = {
                        str(name): [float(x) for x in pose]
                        for name, pose in object_poses_rad.items()
                        if isinstance(pose, list) and len(pose) == 6
                    }
                if isinstance(object_poses_deg, dict):
                    self.last_object_poses_deg = {
                        str(name): [float(x) for x in pose]
                        for name, pose in object_poses_deg.items()
                        if isinstance(pose, list) and len(pose) == 6
                    }
                self.update_object_pose_display()
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
        "  offset 0.0090      设置 mocap z 补偿，方向反了可用 offset -0.0090\n"
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
                        print("[terminal] 用法：offset 0.0090 或 offset -0.0090", flush=True)
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
    key_id = model.key("home").id
    q0_arm = np.array(model.key("home").qpos, copy=True)[qpos_ids]

    object_controls: dict[str, tuple[int, int, int]] = {}
    for object_name, body_name in GRASP_OBJECT_BODY_NAMES.items():
        try:
            body_id = int(model.body(body_name).id)
        except KeyError:
            continue
        jnt_adr = int(model.body_jntadr[body_id])
        jnt_num = int(model.body_jntnum[body_id])
        if jnt_num <= 0:
            continue
        joint_id = jnt_adr
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        object_controls[object_name] = (
            body_id,
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
        )
    shared.update(object_names=list(object_controls))

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

    ee_geom_ids = [x for x in [maybe_geom("ee_axis_x"), maybe_geom("ee_axis_y"), maybe_geom("ee_axis_z")] if x is not None]
    ee_site_ids = [x for x in [maybe_site(SITE_NAME)] if x is not None]
    target_body_id = int(model.body(MOCAP_NAME).id)
    mocap_geom_ids = [int(i) for i in range(model.ngeom) if int(model.geom_bodyid[i]) == target_body_id]
    mocap_site_ids = [int(i) for i in range(model.nsite) if int(model.site_bodyid[i]) == target_body_id]

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

    def current_object_poses() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        poses_rad: dict[str, list[float]] = {}
        poses_deg: dict[str, list[float]] = {}
        for object_name, (body_id, _, _) in object_controls.items():
            pose_rad = np.concatenate([np.array(data.xpos[body_id], copy=True), rpy_from_mat(data.xmat[body_id])])
            pose_deg = np.concatenate([pose_rad[:3], np.rad2deg(pose_rad[3:])])
            poses_rad[object_name] = [float(x) for x in pose_rad]
            poses_deg[object_name] = [float(x) for x in pose_deg]
        return poses_rad, poses_deg

    def set_object_pose(object_name: str, values: list[float] | np.ndarray) -> None:
        if object_name not in object_controls:
            raise ValueError(f"当前模型中没有可移动物体：{object_name}")
        cmd = pose_array_to_command(values)
        _, qpos_adr, qvel_adr = object_controls[object_name]
        data.qpos[qpos_adr:qpos_adr + 3] = cmd.position
        data.qpos[qpos_adr + 3:qpos_adr + 7] = cmd.quat
        data.qvel[qvel_adr:qvel_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

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
        # Streamed trajectory execution: for function_path and move_multi_step, the
        # mocap target advances through sampled points at a fixed interval. This
        # avoids getting stuck forever at the first waypoint when the first point is
        # slightly outside tolerance or orientation is hard to reach.
        stream_waypoints: list[PoseCommand] = []
        stream_index = 0
        stream_last_time = 0.0
        stream_step_time = max(0.005, float(args.function_step_time))
        stream_start_hold_time = max(0.0, float(args.stream_start_hold_time))
        stream_started = False
        stream_wait_until = 0.0
        last_state_time = 0.0
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
                        waypoints = parse_waypoints_rad(command.get("waypoints", []))
                        if waypoints:
                            mode = str(command.get("mode", "move"))
                            motion.active_mode = mode
                            motion.current_goal = waypoints[0]
                            motion.active_goal_index = 1
                            motion.total_goal_count = len(waypoints)
                            set_mocap_pose(motion.current_goal)

                            if mode in {"function_path", "move_multi_step"}:
                                # Function path and UI Multi Step are streamed in time instead
                                # of waiting for each sampled waypoint to be exactly reached.
                                # Before streaming, keep the mocap target at its current pose for
                                # a short hold time, so the robot can first move to the current
                                # visible mocap initial pose instead of the mocap jumping away
                                # immediately when the button is clicked.
                                stream_waypoints = waypoints
                                stream_index = 0
                                stream_started = False
                                stream_wait_until = time.time() + stream_start_hold_time
                                stream_last_time = 0.0
                                motion.current_goal = None
                                motion.active_goal_index = 0
                                motion.remaining_goals = []
                                label = "函数轨迹" if mode == "function_path" else "Multi Step 轨迹"
                                status = (
                                    f"准备执行{label}：mocap 先保持当前位置 {stream_start_hold_time:.2f}s，"
                                    f"机械臂先移动到当前 mocap 初始点；随后每 {stream_step_time:.3f}s 推进一个点"
                                )
                            else:
                                # move_one_step still waits until the target is reached.
                                stream_waypoints = []
                                stream_index = 0
                                motion.remaining_goals = waypoints[1:]
                                status = f"开始执行 {motion.active_mode}: waypoint {motion.active_goal_index}/{motion.total_goal_count}"
                    elif ctype == "stop":
                        stream_waypoints = []
                        stream_index = 0
                        stream_started = False
                        stream_wait_until = 0.0
                        reset_trail_writer()
                        motion.clear()
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
                    elif ctype == "object_pose":
                        object_name = str(command.get("object", ""))
                        set_object_pose(object_name, command.get("pose", []))
                        status = f"已移动 {object_name}: {format_pose(command.get('pose', []))}"
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

            if gripper_actuator_id is not None:
                lo, hi = model.actuator_ctrlrange[gripper_actuator_id]
                data.ctrl[gripper_actuator_id] = float(np.clip(desired_gripper, lo, hi))

            # For streamed function_path / move_multi_step: first hold the current mocap
            # pose for stream_start_hold_time, then send the first sampled waypoint.
            if stream_waypoints and not stream_started:
                now_for_stream = time.time()
                label = "函数轨迹" if motion.active_mode == "function_path" else "Multi Step 轨迹"
                if now_for_stream >= stream_wait_until:
                    stream_started = True
                    stream_index = 0
                    motion.current_goal = stream_waypoints[0]
                    motion.active_goal_index = 1
                    set_mocap_pose(motion.current_goal)
                    stream_last_time = now_for_stream
                    status = f"开始执行{label}：采样点 1/{motion.total_goal_count}"
                else:
                    remain = max(0.0, stream_wait_until - now_for_stream)
                    status = f"{label}准备中：mocap 保持当前初始点，还剩 {remain:.2f}s 后开始轨迹"

            if motion.current_goal is not None:
                pos_err_goal = float(np.linalg.norm(compensated_target_pos() - data.site(site_id).xpos))
                ori_err_goal = orientation_error()

                if motion.active_mode in {"function_path", "move_multi_step"} and stream_waypoints and stream_started:
                    now_for_stream = time.time()
                    label = "函数轨迹" if motion.active_mode == "function_path" else "Multi Step 轨迹"
                    if stream_index < len(stream_waypoints) - 1 and now_for_stream - stream_last_time >= stream_step_time:
                        stream_index += 1
                        motion.current_goal = stream_waypoints[stream_index]
                        motion.active_goal_index = stream_index + 1
                        set_mocap_pose(motion.current_goal)
                        stream_last_time = now_for_stream
                        status = f"{label}执行中：采样点 {motion.active_goal_index}/{motion.total_goal_count}"
                    elif stream_index >= len(stream_waypoints) - 1:
                        # After the final sampled point has been sent, wait until the
                        # robot actually catches up, then mark the path as complete.
                        if pos_err_goal < args.position_tolerance and ori_err_goal < args.orientation_tolerance:
                            status = f"{label}执行完成。"
                            stream_waypoints = []
                            stream_index = 0
                            stream_started = False
                            stream_wait_until = 0.0
                            motion.clear()
                else:
                    # Normal multi-step execution waits until each waypoint is reached.
                    if pos_err_goal < args.position_tolerance and ori_err_goal < args.orientation_tolerance:
                        if motion.remaining_goals:
                            motion.current_goal = motion.remaining_goals.pop(0)
                            motion.active_goal_index += 1
                            set_mocap_pose(motion.current_goal)
                            status = f"继续执行 waypoint {motion.active_goal_index}/{motion.total_goal_count}"
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
                object_poses_rad, object_poses_deg = current_object_poses()
                shared.update(
                    pose_rad=[float(x) for x in pose_rad],
                    pose_deg=[float(x) for x in pose_deg],
                    pos_err=float(np.linalg.norm(compensated_target_pos() - data.site(site_id).xpos)),
                    ori_err=orientation_error(),
                    status=status,
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
                    object_names=list(object_controls),
                    object_poses_rad=object_poses_rad,
                    object_poses_deg=object_poses_deg,
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
    parser.add_argument("--mocap-z-comp", type=float, default=0.0090, help="Fixed Z compensation between visible mocap axes and controlled EE site. Use -0.0090 if direction is reversed.")
    parser.add_argument("--function-step-time", type=float, default=0.06, help="Time interval between sampled points during direct function-path and streamed Multi Step execution. Smaller is faster. Default: 0.06s")
    parser.add_argument("--stream-start-hold-time", type=float, default=2.0, help="Before streaming function_path or Multi Step, keep mocap at its current pose for this many seconds. Default: 1.0s")
    parser.add_argument("--no-terminal-commands", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ui_client:
        return run_ui_client(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
