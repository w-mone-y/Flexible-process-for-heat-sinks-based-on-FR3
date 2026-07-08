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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import numpy as np


SITE_NAME = "attachment_site"
MOCAP_NAME = "target"
JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
ACTUATOR_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
GRIPPER_ACTUATOR_NAME = "gripper"
WRIST_SPIN_JOINT_NAME = "wrist_spin"
WRIST_SPIN_ACTUATOR_NAME = "wrist_spin"
OBJECT_OFFSET_IN_EE = np.asarray([0.0, 0.0, 0.17], dtype=float)
GRIPPER_CLOSED_CTRL = 0.0


@dataclass(frozen=True)
class PoseCommand:
    position: np.ndarray
    quat: np.ndarray  # MuJoCo quaternion order [w, x, y, z].
    raw_pose: np.ndarray  # display/debug pose. For quaternion commands: x y z qw qx qy qz.


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


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
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


def mat_from_quat(quat: np.ndarray) -> np.ndarray:
    import mujoco

    mat = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def normalize_quat(quat: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise ValueError("quat must have 4 elements: qw qx qy qz")
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("quaternion norm is too small")
    q = q / n
    # Keep a stable display convention. q and -q are the same attitude.
    if q[0] < 0.0:
        q = -q
    return q


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)
    return normalize_quat(
        np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
    )


def quat_from_axis_angle(axis: list[float] | np.ndarray, angle_rad: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis_arr))
    if n < 1e-12:
        raise ValueError("axis norm is too small")
    axis_arr = axis_arr / n
    half = float(angle_rad) * 0.5
    return normalize_quat(np.concatenate([[math.cos(half)], axis_arr * math.sin(half)]))


def pose_array_to_command(values: list[float] | np.ndarray) -> PoseCommand:
    pose = np.asarray(values, dtype=float)
    if pose.shape == (6,):
        quat = quat_from_rpy(float(pose[3]), float(pose[4]), float(pose[5]))
        raw_pose = np.concatenate([pose[:3], quat])
        return PoseCommand(position=pose[:3].copy(), quat=quat, raw_pose=raw_pose)
    if pose.shape == (7,):
        quat = normalize_quat(pose[3:7])
        return PoseCommand(position=pose[:3].copy(), quat=quat, raw_pose=np.concatenate([pose[:3], quat]))
    raise ValueError("pose must have 6 elements x y z roll pitch yaw or 7 elements x y z qw qx qy qz")


def pose_command_from_position_quat(position: np.ndarray, quat: np.ndarray) -> PoseCommand:
    mat = mat_from_quat(quat)
    raw_pose = np.concatenate([np.asarray(position, dtype=float), rpy_from_mat(mat)])
    return PoseCommand(
        position=np.asarray(position, dtype=float).copy(),
        quat=np.asarray(quat, dtype=float).copy(),
        raw_pose=raw_pose,
    )


def format_pose(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(x):.6f}" for x in values)


def normalize_vec(vec: list[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("vector norm is too small")
    return v / n


def world_direction_from_name(name: str) -> np.ndarray:
    key = str(name).strip().lower()
    mapping = {
        "down": np.asarray([0.0, 0.0, -1.0]),
        "朝下": np.asarray([0.0, 0.0, -1.0]),
        "up": np.asarray([0.0, 0.0, 1.0]),
        "朝上": np.asarray([0.0, 0.0, 1.0]),
        "front": np.asarray([1.0, 0.0, 0.0]),
        "forward": np.asarray([1.0, 0.0, 0.0]),
        "朝前": np.asarray([1.0, 0.0, 0.0]),
        "back": np.asarray([-1.0, 0.0, 0.0]),
        "朝后": np.asarray([-1.0, 0.0, 0.0]),
        "left": np.asarray([0.0, 1.0, 0.0]),
        "朝左": np.asarray([0.0, 1.0, 0.0]),
        "right": np.asarray([0.0, -1.0, 0.0]),
        "朝右": np.asarray([0.0, -1.0, 0.0]),
    }
    if key not in mapping:
        raise ValueError(f"unknown direction: {name}")
    return mapping[key].copy()


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self.pose_rad = [0.0] * 6
        self.pose_deg = [0.0] * 6
        self.pose_quat = [1.0, 0.0, 0.0, 0.0]
        self.object_pose_quat = [1.0, 0.0, 0.0, 0.0]
        self.object_pose_rad = [0.0] * 6
        self.object_pose_deg = [0.0] * 6
        self.joint_angles_rad = [0.0] * 7
        self.joint_angles_deg = [0.0] * 7
        self.joint_limit_margin = 0.0
        self.joint_center_cost = 0.0
        self.joint_centering_active = True
        self.x_spin_ready_score = 0.0
        self.pos_err = 0.0
        self.ori_err = 0.0
        self.status = "starting"
        self.show_ee_axes = True
        self.show_mocap = True
        self.mocap_z_comp = 0.0
        self.viewer_running = False
        self.waypoint_index = 0
        self.waypoint_count = 0
        self.active_mode = "idle"
        self.server_time = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "pose_rad": list(self.pose_rad),
                "pose_deg": list(self.pose_deg),
                "pose_quat": list(self.pose_quat),
                "object_pose_quat": list(self.object_pose_quat),
                "object_pose_rad": list(self.object_pose_rad),
                "object_pose_deg": list(self.object_pose_deg),
                "joint_angles_rad": list(self.joint_angles_rad),
                "joint_angles_deg": list(self.joint_angles_deg),
                "joint_limit_margin": float(self.joint_limit_margin),
                "joint_center_cost": float(self.joint_center_cost),
                "joint_centering_active": bool(self.joint_centering_active),
                "x_spin_ready_score": float(self.x_spin_ready_score),
                "pos_err": float(self.pos_err),
                "ori_err": float(self.ori_err),
                "status": str(self.status),
                "show_ee_axes": bool(self.show_ee_axes),
                "show_mocap": bool(self.show_mocap),
                "mocap_z_comp": float(self.mocap_z_comp),
                "viewer_running": bool(self.viewer_running),
                "waypoint_index": int(self.waypoint_index),
                "waypoint_count": int(self.waypoint_count),
                "active_mode": str(self.active_mode),
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
            if self.path == "/move":
                waypoints = payload.get("waypoints", [])
                mode = str(payload.get("mode", "move_one_step"))
                if mode != "move_one_step":
                    raise ValueError("base_develop only supports move_one_step")
                if not isinstance(waypoints, list) or len(waypoints) != 1:
                    raise ValueError("move_one_step requires exactly one waypoint")
                self.shared.commands.put({"type": "move", "mode": mode, "waypoints": waypoints})
                self._send_json({"ok": True})
            elif self.path == "/object_move":
                waypoint = payload.get("waypoint", None)
                waypoints = payload.get("waypoints", None)
                if waypoint is None and isinstance(waypoints, list) and waypoints:
                    waypoint = waypoints[0]
                if waypoint is None:
                    raise ValueError("object_move requires waypoint: x y z qw qx qy qz")
                optimize_joints = bool(payload.get("optimize_joints", True))
                self.shared.commands.put({"type": "object_move", "waypoint": waypoint, "optimize_joints": optimize_joints})
                self._send_json({"ok": True, "optimize_joints": optimize_joints})
            elif self.path == "/object_preset":
                direction = str(payload.get("direction", ""))
                optimize_joints = bool(payload.get("optimize_joints", True))
                self.shared.commands.put({"type": "object_preset", "direction": direction, "optimize_joints": optimize_joints})
                self._send_json({"ok": True, "direction": direction, "optimize_joints": optimize_joints})
            elif self.path == "/object_rotate_body":
                axis = str(payload.get("axis", ""))
                angle_deg = float(payload.get("angle_deg", 0.0))
                optimize_joints = bool(payload.get("optimize_joints", True))
                self.shared.commands.put({"type": "object_rotate_body", "axis": axis, "angle_deg": angle_deg, "optimize_joints": optimize_joints})
                self._send_json({"ok": True, "axis": axis, "angle_deg": angle_deg, "optimize_joints": optimize_joints})
            elif self.path == "/object_prepare_x_spin":
                seconds = float(payload.get("seconds", 0.0)) if "seconds" in payload else None
                self.shared.commands.put({"type": "object_prepare_x_spin", "seconds": seconds})
                self._send_json({"ok": True, "seconds": seconds})
            elif self.path == "/stop":
                self.shared.commands.put({"type": "stop"})
                self._send_json({"ok": True})
            elif self.path == "/ee_axes":
                visible = bool(payload.get("visible", True))
                self.shared.update(show_ee_axes=visible)
                self.shared.commands.put({"type": "ee_axes", "visible": visible})
                self._send_json({"ok": True, "visible": visible})
            elif self.path == "/mocap":
                visible = bool(payload.get("visible", True))
                self.shared.update(show_mocap=visible)
                self.shared.commands.put({"type": "mocap", "visible": visible})
                self._send_json({"ok": True, "visible": visible})
            elif self.path == "/offset":
                value = float(payload.get("mocap_z_comp", 0.0))
                self.shared.update(mocap_z_comp=value)
                self.shared.commands.put({"type": "offset", "mocap_z_comp": value})
                self._send_json({"ok": True, "mocap_z_comp": value})
            elif self.path == "/quit":
                self.shared.commands.put({"type": "quit"})
                self._send_json({"ok": True})
            else:
                self._send_json({"ok": False, "error": "not found"}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)


def start_http_server(shared: SharedState, host: str, port: int) -> ThreadingHTTPServer:
    handler_cls = type("BaseDevelopRequestHandler", (RequestHandler,), {"shared": shared})
    server = ThreadingHTTPServer((host, port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
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


def launch_ui_subprocess(script_path: Path, port: int, ui_python: str) -> subprocess.Popen[Any]:
    cmd = [ui_python, str(script_path), "--ui-client", "--port", str(port)]
    print("[UI] " + " ".join(cmd), flush=True)
    return subprocess.Popen(cmd)


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
        from PySide6.QtCore import QEvent, Qt, QTimer
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
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
            self.setWindowTitle("base_develop - FR3 mocap follow")
            self.last_pose_rad = [0.0] * 6
            self.last_pose_deg = [0.0] * 6
            self.last_pose_quat = [1.0, 0.0, 0.0, 0.0]
            self.last_object_pose_rad = [0.0] * 6
            self.last_object_pose_deg = [0.0] * 6
            self.last_object_pose_quat = [1.0, 0.0, 0.0, 0.0]

            layout = QVBoxLayout(self)

            state_group = QGroupBox("Current End Effector Pose")
            state_layout = QVBoxLayout(state_group)
            self.pose_rad_label = QLabel("waiting...")
            self.pose_deg_label = QLabel("waiting...")
            self.pose_quat_label = QLabel("quat: waiting...")
            self.object_pose_label = QLabel("object: waiting...")
            self.object_quat_label = QLabel("object quat: waiting...")
            self.joint_deg_label = QLabel("Joints deg: waiting...")
            self.joint_rad_label = QLabel("Joints rad: waiting...")
            self.joint_margin_label = QLabel("joint margin: waiting...")
            self.err_label = QLabel("pos_err: --   ori_err: --")
            self.status_label = QLabel("starting...")
            for label in (
                self.pose_rad_label,
                self.pose_deg_label,
                self.pose_quat_label,
                self.object_pose_label,
                self.object_quat_label,
                self.err_label,
                self.status_label,
            ):
                self.make_label_copyable(label)
                state_layout.addWidget(label)
            layout.addWidget(state_group)

            joint_group = QGroupBox("Seven FR3 Joint Angles / 七轴机械臂关节角")
            joint_layout = QVBoxLayout(joint_group)
            for label in (self.joint_deg_label, self.joint_rad_label, self.joint_margin_label):
                self.make_label_copyable(label)
                joint_layout.addWidget(label)
            layout.addWidget(joint_group)

            move_group = QGroupBox("Move One Step")
            move_layout = QVBoxLayout(move_group)
            unit_row = QHBoxLayout()
            unit_row.addWidget(QLabel("角度单位:"))
            self.unit_combo = QComboBox()
            self.unit_combo.addItems(["rad", "deg"])
            self.unit_combo.setCurrentText("deg")
            unit_row.addWidget(self.unit_combo)
            unit_row.addStretch(1)
            move_layout.addLayout(unit_row)

            self.pose_boxes: dict[str, QDoubleSpinBox] = {}
            grid = QGridLayout()
            defaults = {"x": 0.45, "y": 0.0, "z": 0.45, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}
            for col, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
                box = QDoubleSpinBox()
                box.setDecimals(6)
                box.setRange(-10000.0, 10000.0)
                box.setSingleStep(0.01 if name in {"x", "y", "z"} else 1.0)
                box.setValue(defaults[name])
                self.pose_boxes[name] = box
                grid.addWidget(QLabel(name), 0, col)
                grid.addWidget(box, 1, col)
            move_layout.addLayout(grid)

            array_row = QHBoxLayout()
            self.pose_array_input = QLineEdit()
            self.pose_array_input.setPlaceholderText("[x, y, z, roll, pitch, yaw]")
            array_row.addWidget(QLabel("数组:"))
            array_row.addWidget(self.pose_array_input, 1)
            fill_array_btn = QPushButton("从数组填入")
            fill_array_btn.clicked.connect(self.fill_from_array)
            array_row.addWidget(fill_array_btn)
            move_layout.addLayout(array_row)

            btn_row = QHBoxLayout()
            move_btn = QPushButton("移动到该位姿")
            move_btn.clicked.connect(self.move_one_step)
            fill_current_btn = QPushButton("把当前位姿填入")
            fill_current_btn.clicked.connect(self.fill_current_pose)
            stop_btn = QPushButton("停止")
            stop_btn.clicked.connect(lambda: self.post("/stop", {}))
            btn_row.addWidget(move_btn)
            btn_row.addWidget(fill_current_btn)
            btn_row.addWidget(stop_btn)
            btn_row.addStretch(1)
            move_layout.addLayout(btn_row)
            layout.addWidget(move_group)

            object_group = QGroupBox("Object Coordinate Control / 待抓取物体坐标系控制")
            object_layout = QVBoxLayout(object_group)

            opt_row = QHBoxLayout()
            self.object_optimize_checkbox = QCheckBox("全局优化关节角度：所有 mocap 跟随 / Move One Step / 物体控制都启用")
            self.object_optimize_checkbox.setChecked(True)
            self.object_optimize_checkbox.setEnabled(False)
            opt_row.addWidget(self.object_optimize_checkbox)
            opt_row.addStretch(1)
            object_layout.addLayout(opt_row)

            prepare_row = QHBoxLayout()
            prepare_x_btn = QPushButton("预优化：最大化绕物体X轴范围")
            prepare_x_btn.clicked.connect(self.object_prepare_x_spin)
            prepare_row.addWidget(prepare_x_btn)
            prepare_row.addWidget(QLabel("保持当前物体位姿，先调整七轴到更适合绕自身X轴旋转的构型"))
            prepare_row.addStretch(1)
            object_layout.addLayout(prepare_row)

            preset_row = QHBoxLayout()
            preset_row.addWidget(QLabel("保持物体当前位置，让物体蓝色Z轴朝向:"))
            for text, direction in (("朝下", "down"), ("朝前", "front"), ("朝左", "left"), ("朝右", "right"), ("朝后", "back"), ("朝上", "up")):
                btn = QPushButton(text)
                btn.clicked.connect(lambda _checked=False, d=direction: self.object_preset(d))
                preset_row.addWidget(btn)
            preset_row.addStretch(1)
            object_layout.addLayout(preset_row)

            obj_rot_row = QHBoxLayout()
            obj_rot_row.addWidget(QLabel("绕物体自身轴旋转 [-180°, 180°]:"))
            self.object_rot_boxes: dict[str, QDoubleSpinBox] = {}
            for axis in ("x", "y", "z"):
                box = QDoubleSpinBox()
                box.setDecimals(3)
                box.setRange(-180.0, 180.0)
                box.setSingleStep(5.0)
                box.setValue(0.0)
                self.object_rot_boxes[axis] = box
                obj_rot_row.addWidget(QLabel(axis.upper()))
                obj_rot_row.addWidget(box)
                btn = QPushButton(f"绕自身{axis.upper()}旋转")
                btn.clicked.connect(lambda _checked=False, a=axis: self.object_rotate_body(a))
                obj_rot_row.addWidget(btn)
            obj_rot_row.addStretch(1)
            object_layout.addLayout(obj_rot_row)

            obj_move_group = QGroupBox("Object Move One Step / 物体 Move One Step")
            obj_move_layout = QVBoxLayout(obj_move_group)
            self.object_pose_boxes: dict[str, QDoubleSpinBox] = {}
            obj_grid = QGridLayout()
            obj_defaults = {"x": 0.45, "y": 0.0, "z": 0.62, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
            for col, name in enumerate(("x", "y", "z", "qw", "qx", "qy", "qz")):
                box = QDoubleSpinBox()
                box.setDecimals(6)
                box.setRange(-10000.0, 10000.0)
                box.setSingleStep(0.01 if name in {"x", "y", "z"} else 0.01)
                box.setValue(obj_defaults[name])
                self.object_pose_boxes[name] = box
                obj_grid.addWidget(QLabel(name), 0, col)
                obj_grid.addWidget(box, 1, col)
            obj_move_layout.addLayout(obj_grid)

            obj_array_row = QHBoxLayout()
            self.object_pose_array_input = QLineEdit()
            self.object_pose_array_input.setPlaceholderText("[x, y, z, qw, qx, qy, qz]")
            obj_array_row.addWidget(QLabel("物体数组:"))
            obj_array_row.addWidget(self.object_pose_array_input, 1)
            obj_fill_array_btn = QPushButton("从数组填入")
            obj_fill_array_btn.clicked.connect(self.fill_object_from_array)
            obj_array_row.addWidget(obj_fill_array_btn)
            obj_move_layout.addLayout(obj_array_row)

            obj_btn_row = QHBoxLayout()
            obj_move_btn = QPushButton("移动物体到该位姿")
            obj_move_btn.clicked.connect(self.object_move_one_step)
            obj_fill_current_btn = QPushButton("把当前物体位姿填入")
            obj_fill_current_btn.clicked.connect(self.fill_current_object_pose)
            obj_norm_btn = QPushButton("归一化物体四元数")
            obj_norm_btn.clicked.connect(self.normalize_object_quat_boxes)
            obj_btn_row.addWidget(obj_move_btn)
            obj_btn_row.addWidget(obj_fill_current_btn)
            obj_btn_row.addWidget(obj_norm_btn)
            obj_btn_row.addStretch(1)
            obj_move_layout.addLayout(obj_btn_row)
            object_layout.addWidget(obj_move_group)
            layout.addWidget(object_group)

            vis_group = QGroupBox("Visibility")
            vis_layout = QHBoxLayout(vis_group)
            self.ee_checkbox = QCheckBox("显示末端坐标轴")
            self.ee_checkbox.setChecked(True)
            self.ee_checkbox.toggled.connect(lambda value: self.post("/ee_axes", {"visible": bool(value)}))
            self.mocap_checkbox = QCheckBox("显示 mocap target")
            self.mocap_checkbox.setChecked(True)
            self.mocap_checkbox.toggled.connect(lambda value: self.post("/mocap", {"visible": bool(value)}))
            vis_layout.addWidget(self.ee_checkbox)
            vis_layout.addWidget(self.mocap_checkbox)
            vis_layout.addStretch(1)
            layout.addWidget(vis_group)

            # Make every displayed text and numeric/text input copy-friendly.
            # QLabel needs explicit selectable flags; QDoubleSpinBox owns an internal
            # QLineEdit, so we configure both the spinbox and its editor.
            self.make_all_text_widgets_copyable()

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh_state)
            self.timer.start(80)

        def make_label_copyable(self, label: QLabel) -> None:
            label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            label.setFocusPolicy(Qt.StrongFocus)

        def make_line_edit_copyable(self, edit: QLineEdit) -> None:
            # QLineEdit already supports selection/copy by default, but these flags
            # make the behavior consistent for array inputs and QDoubleSpinBox editors.
            edit.setFocusPolicy(Qt.StrongFocus)
            edit.setDragEnabled(True)
            edit.setContextMenuPolicy(Qt.DefaultContextMenu)
            edit.installEventFilter(self)

        def make_number_box_copyable(self, box: QDoubleSpinBox) -> None:
            box.setFocusPolicy(Qt.StrongFocus)
            box.setKeyboardTracking(True)
            self.make_line_edit_copyable(box.lineEdit())

        def make_all_text_widgets_copyable(self) -> None:
            for label in self.findChildren(QLabel):
                self.make_label_copyable(label)
            for edit in self.findChildren(QLineEdit):
                self.make_line_edit_copyable(edit)
            for box in self.findChildren(QDoubleSpinBox):
                self.make_number_box_copyable(box)

        def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
            # When a text/numeric field gets focus, select the current value after Qt
            # finishes its own focus handling. This makes Cmd/Ctrl+C work immediately.
            if event.type() == QEvent.FocusIn and isinstance(obj, QLineEdit):
                QTimer.singleShot(0, obj.selectAll)
            return super().eventFilter(obj, event)

        def unit(self) -> str:
            return "deg" if self.unit_combo.currentText() == "deg" else "rad"

        def post(self, path: str, payload: dict[str, Any]) -> None:
            try:
                post_json(base_url + path, payload, timeout=0.5)
            except Exception as exc:
                self.status_label.setText(f"请求失败: {exc}")

        def get_pose_rad(self) -> list[float]:
            values = [self.pose_boxes[name].value() for name in ("x", "y", "z", "roll", "pitch", "yaw")]
            if self.unit() == "deg":
                values[3:] = [math.radians(value) for value in values[3:]]
            return [float(value) for value in values]

        def parse_array(self, text: str) -> list[float]:
            raw = text.strip().replace("，", ",")
            if len(raw) >= 2 and raw[0] in "[(" and raw[-1] in ")]":
                raw = raw[1:-1]
            fields = [part for part in raw.replace(",", " ").split() if part]
            if len(fields) != 6:
                raise ValueError("请输入 6 个数值: x y z roll pitch yaw")
            return [float(part) for part in fields]

        def fill_from_array(self) -> None:
            try:
                values = self.parse_array(self.pose_array_input.text())
            except ValueError as exc:
                QMessageBox.warning(self, "数组格式错误", str(exc))
                return
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), values, strict=True):
                self.pose_boxes[name].setValue(float(value))

        def fill_current_pose(self) -> None:
            values = list(self.last_pose_deg if self.unit() == "deg" else self.last_pose_rad)
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), values, strict=True):
                self.pose_boxes[name].setValue(float(value))

        def move_one_step(self) -> None:
            self.post("/move", {"mode": "move_one_step", "waypoints": [self.get_pose_rad()]})

        def parse_object_array(self, text: str) -> list[float]:
            raw = text.strip().replace("，", ",")
            if len(raw) >= 2 and raw[0] in "[(" and raw[-1] in ")]":
                raw = raw[1:-1]
            fields = [part for part in raw.replace(",", " ").split() if part]
            if len(fields) != 7:
                raise ValueError("请输入 7 个数值: x y z qw qx qy qz")
            return [float(part) for part in fields]

        def fill_object_from_array(self) -> None:
            try:
                values = self.parse_object_array(self.object_pose_array_input.text())
            except ValueError as exc:
                QMessageBox.warning(self, "物体数组格式错误", str(exc))
                return
            for name, value in zip(("x", "y", "z", "qw", "qx", "qy", "qz"), values, strict=True):
                self.object_pose_boxes[name].setValue(float(value))

        def normalize_object_quat_boxes(self) -> None:
            q = [self.object_pose_boxes[name].value() for name in ("qw", "qx", "qy", "qz")]
            n = math.sqrt(sum(float(x) * float(x) for x in q))
            if n <= 1e-12:
                QMessageBox.warning(self, "四元数错误", "四元数模长太小，无法归一化")
                return
            q = [float(x) / n for x in q]
            if q[0] < 0.0:
                q = [-x for x in q]
            for name, value in zip(("qw", "qx", "qy", "qz"), q, strict=True):
                self.object_pose_boxes[name].setValue(value)

        def get_object_pose_quat(self) -> list[float]:
            values = [self.object_pose_boxes[name].value() for name in ("x", "y", "z", "qw", "qx", "qy", "qz")]
            q = values[3:7]
            n = math.sqrt(sum(float(x) * float(x) for x in q))
            if n <= 1e-12:
                raise ValueError("物体四元数模长太小")
            values[3:7] = [float(x) / n for x in q]
            if values[3] < 0.0:
                values[3:7] = [-float(x) for x in values[3:7]]
            return [float(value) for value in values]

        def fill_current_object_pose(self) -> None:
            values = list(self.last_object_pose_rad[:3]) + list(self.last_object_pose_quat)
            for name, value in zip(("x", "y", "z", "qw", "qx", "qy", "qz"), values, strict=True):
                self.object_pose_boxes[name].setValue(float(value))

        def object_move_one_step(self) -> None:
            try:
                waypoint = self.get_object_pose_quat()
            except ValueError as exc:
                QMessageBox.warning(self, "物体位姿错误", str(exc))
                return
            self.post("/object_move", {"waypoint": waypoint, "optimize_joints": bool(self.object_optimize_checkbox.isChecked())})

        def object_preset(self, direction: str) -> None:
            self.post("/object_preset", {"direction": direction, "optimize_joints": bool(self.object_optimize_checkbox.isChecked())})

        def object_prepare_x_spin(self) -> None:
            self.post("/object_prepare_x_spin", {})

        def object_rotate_body(self, axis: str) -> None:
            angle = float(self.object_rot_boxes[axis].value())
            self.post(
                "/object_rotate_body",
                {"axis": axis, "angle_deg": angle, "optimize_joints": bool(self.object_optimize_checkbox.isChecked())},
            )

        def refresh_state(self) -> None:
            try:
                state = get_json(base_url + "/state", timeout=0.5)
                self.last_pose_rad = [float(x) for x in state.get("pose_rad", [0.0] * 6)]
                self.last_pose_deg = [float(x) for x in state.get("pose_deg", [0.0] * 6)]
                self.last_pose_quat = [float(x) for x in state.get("pose_quat", [1.0, 0.0, 0.0, 0.0])]
                self.last_object_pose_rad = [float(x) for x in state.get("object_pose_rad", [0.0] * 6)]
                self.last_object_pose_deg = [float(x) for x in state.get("object_pose_deg", [0.0] * 6)]
                self.last_object_pose_quat = [float(x) for x in state.get("object_pose_quat", [1.0, 0.0, 0.0, 0.0])]
                self.pose_rad_label.setText("EE rad: " + format_pose(self.last_pose_rad))
                self.pose_deg_label.setText("EE deg: " + format_pose(self.last_pose_deg))
                self.pose_quat_label.setText("EE quat [qw qx qy qz]: " + format_pose(self.last_pose_quat))
                self.object_pose_label.setText("Object deg: " + format_pose(self.last_object_pose_deg))
                self.object_quat_label.setText("Object quat [qw qx qy qz]: " + format_pose(self.last_object_pose_quat))
                joint_deg = [float(x) for x in state.get("joint_angles_deg", [0.0] * 7)]
                joint_rad = [float(x) for x in state.get("joint_angles_rad", [0.0] * 7)]
                self.joint_deg_label.setText(
                    "Joints deg: " + "  ".join(f"J{i+1}:{joint_deg[i]: .2f}°" for i in range(7))
                )
                self.joint_rad_label.setText(
                    "Joints rad: " + "  ".join(f"J{i+1}:{joint_rad[i]: .4f}" for i in range(7))
                )
                self.joint_margin_label.setText(
                    f"joint min-margin: {float(state.get('joint_limit_margin', 0.0)):.3f}   "
                    f"center-cost: {float(state.get('joint_center_cost', 0.0)):.3f}   "
                    f"x-spin-ready: {float(state.get('x_spin_ready_score', 0.0)):.3f}   "
                    f"centering: {'ON' if bool(state.get('joint_centering_active', False)) else 'OFF'}"
                )
                self.err_label.setText(
                    f"pos_err: {float(state.get('pos_err', 0.0)):.5f} m   "
                    f"ori_err: {float(state.get('ori_err', 0.0)):.5f} rad"
                )
                mode = str(state.get("active_mode", "idle"))
                wp_i = int(state.get("waypoint_index", 0))
                wp_n = int(state.get("waypoint_count", 0))
                self.status_label.setText(f"{state.get('status', '')} [{mode} {wp_i}/{wp_n}]")

                show_ee = bool(state.get("show_ee_axes", True))
                if self.ee_checkbox.isChecked() != show_ee:
                    self.ee_checkbox.blockSignals(True)
                    self.ee_checkbox.setChecked(show_ee)
                    self.ee_checkbox.blockSignals(False)
                show_mocap = bool(state.get("show_mocap", True))
                if self.mocap_checkbox.isChecked() != show_mocap:
                    self.mocap_checkbox.blockSignals(True)
                    self.mocap_checkbox.setChecked(show_mocap)
                    self.mocap_checkbox.blockSignals(False)
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
    panel.resize(1280, 720)
    panel.show()
    print("[UI] base_develop PySide6 panel opened.", flush=True)
    return int(app.exec())


def print_terminal_help() -> None:
    print(
        "\n[终端命令]\n"
        "  ee_axes on/off      显示/隐藏机械臂末端坐标轴\n"
        "  mocap on/off        显示/隐藏 mocap target\n"
        "  offset 0.0          设置 mocap z 补偿\n"
        "  object_preset down/front/left/right/back/up   保持物体当前位置，改变物体蓝轴朝向\n"
        "  object_rot x/y/z 角度                         绕待抓取物体自身轴旋转，单位 deg\n"
        "  prepare_x_spin                                保持当前物体位姿，预优化七轴以扩大绕物体X轴旋转范围\n"
        "  object_moveq x y z qw qx qy qz                物体 Move One Step，MuJoCo 四元数\n"
        "  stop                停止 Move One Step\n"
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
            if cmd in {"ee_axes on", "ee on"}:
                shared.show_ee_axes = True
                shared.commands.put({"type": "ee_axes", "visible": True})
            elif cmd in {"ee_axes off", "ee off"}:
                shared.show_ee_axes = False
                shared.commands.put({"type": "ee_axes", "visible": False})
            elif cmd in {"mocap on", "target on"}:
                shared.show_mocap = True
                shared.commands.put({"type": "mocap", "visible": True})
            elif cmd in {"mocap off", "target off"}:
                shared.show_mocap = False
                shared.commands.put({"type": "mocap", "visible": False})
            elif cmd.startswith("offset "):
                try:
                    value = float(cmd.split(maxsplit=1)[1])
                except ValueError:
                    print("[terminal] offset 后面需要数字", flush=True)
                    continue
                shared.mocap_z_comp = value
                shared.commands.put({"type": "offset", "mocap_z_comp": value})
            elif cmd.startswith("object_preset "):
                direction = cmd.split(maxsplit=1)[1].strip()
                shared.commands.put({"type": "object_preset", "direction": direction})
            elif cmd.startswith("object_rot "):
                parts = cmd.split()
                if len(parts) != 3 or parts[1] not in {"x", "y", "z"}:
                    print("[terminal] 用法: object_rot x/y/z angle_deg", flush=True)
                    continue
                try:
                    angle = float(parts[2])
                except ValueError:
                    print("[terminal] angle_deg 需要数字", flush=True)
                    continue
                shared.commands.put({"type": "object_rotate_body", "axis": parts[1], "angle_deg": angle})
            elif cmd == "prepare_x_spin":
                shared.commands.put({"type": "object_prepare_x_spin"})
            elif cmd.startswith("object_moveq "):
                parts = cmd.split()[1:]
                if len(parts) != 7:
                    print("[terminal] 用法: object_moveq x y z qw qx qy qz", flush=True)
                    continue
                try:
                    waypoint = [float(x) for x in parts]
                except ValueError:
                    print("[terminal] object_moveq 后面需要 7 个数字", flush=True)
                    continue
                shared.commands.put({"type": "object_move", "waypoint": waypoint})
            elif cmd == "stop":
                shared.commands.put({"type": "stop"})
            elif cmd == "help":
                print_terminal_help()

    threading.Thread(target=loop, daemon=True).start()


def run_controller(args: argparse.Namespace) -> int:
    import mujoco
    import mujoco.viewer

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"找不到 XML 文件: {xml_path}")

    shared = SharedState()
    shared.show_ee_axes = not bool(args.hide_ee_axes)
    shared.show_mocap = not bool(args.hide_mocap)
    shared.mocap_z_comp = float(args.mocap_z_comp)

    server = start_http_server(shared, "127.0.0.1", int(args.port))
    actual_port = int(server.server_address[1])
    print(f"[1/5] HTTP control server: http://127.0.0.1:{actual_port}", flush=True)

    if not args.no_terminal_commands:
        start_terminal_thread(shared)

    ui_proc: Optional[subprocess.Popen[Any]] = None
    if not args.no_ui:
        ui_proc = launch_ui_subprocess(Path(__file__).resolve(), actual_port, find_python_for_ui(args.ui_python))

    print(f"[2/5] Loading XML: {xml_path}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    original_geom_rgba = model.geom_rgba.copy()
    original_site_rgba = model.site_rgba.copy()
    model.opt.timestep = float(args.dt)
    if not args.real_gravity:
        model.opt.gravity[:] = 0.0

    site_id = int(model.site(SITE_NAME).id)
    mocap_id = int(model.body(MOCAP_NAME).mocapid[0])
    if mocap_id < 0:
        raise ValueError(f"Body {MOCAP_NAME!r} is not a mocap body.")

    joint_ids = np.asarray([int(model.joint(name).id) for name in JOINT_NAMES], dtype=int)
    qpos_ids = np.asarray([int(model.jnt_qposadr[jid]) for jid in joint_ids], dtype=int)
    dof_ids = np.asarray([int(model.jnt_dofadr[jid]) for jid in joint_ids], dtype=int)
    actuator_ids = np.asarray([int(model.actuator(name).id) for name in ACTUATOR_NAMES], dtype=int)

    model.body_gravcomp[:] = 0.0
    if args.gravity_comp:
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

    gripper_actuator_id: Optional[int]
    try:
        gripper_actuator_id = int(model.actuator(GRIPPER_ACTUATOR_NAME).id)
    except KeyError:
        gripper_actuator_id = None

    wrist_spin_joint_id: Optional[int]
    wrist_spin_qpos_id: Optional[int]
    wrist_spin_actuator_id: Optional[int]
    try:
        wrist_spin_joint_id = int(model.joint(WRIST_SPIN_JOINT_NAME).id)
        wrist_spin_qpos_id = int(model.jnt_qposadr[wrist_spin_joint_id])
        wrist_spin_actuator_id = int(model.actuator(WRIST_SPIN_ACTUATOR_NAME).id)
    except KeyError:
        wrist_spin_joint_id = None
        wrist_spin_qpos_id = None
        wrist_spin_actuator_id = None

    key_id = int(model.key("home").id)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    q0_arm = np.array(data.qpos[qpos_ids], copy=True)
    joint_lower = np.full(len(joint_ids), -np.inf, dtype=float)
    joint_upper = np.full(len(joint_ids), np.inf, dtype=float)
    joint_mid = q0_arm.copy()
    joint_half_range = np.ones(len(joint_ids), dtype=float)
    for i, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            joint_lower[i] = float(lo)
            joint_upper[i] = float(hi)
            joint_mid[i] = 0.5 * (float(lo) + float(hi))
            joint_half_range[i] = max(0.5 * (float(hi) - float(lo)), 1e-6)

    def joint_centering_metrics(q_arm: np.ndarray) -> tuple[float, float]:
        limited = np.isfinite(joint_lower) & np.isfinite(joint_upper)
        if not np.any(limited):
            return 1.0, 0.0
        lo = joint_lower[limited]
        hi = joint_upper[limited]
        q_lim = np.asarray(q_arm, dtype=float)[limited]
        span = np.maximum(hi - lo, 1e-9)
        margin = np.minimum(q_lim - lo, hi - q_lim) / span
        normalized_center_error = (np.asarray(q_arm, dtype=float) - joint_mid) / joint_half_range
        cost = float(np.mean(normalized_center_error[limited] ** 2))
        return float(np.min(margin)), cost

    def joint_limit_avoidance_velocity(q_arm: np.ndarray) -> np.ndarray:
        # Secondary velocity used only inside the 5D nullspace. It combines a gentle
        # pull toward joint centers with a stronger barrier-like repulsion near limits.
        q_arm = np.asarray(q_arm, dtype=float)
        center_velocity = float(args.joint_centering_gain) * (joint_mid - q_arm)
        if float(args.joint_limit_barrier_gain) <= 0.0:
            return center_velocity
        eta = (q_arm - joint_mid) / np.maximum(joint_half_range, 1e-8)
        eta = np.clip(eta, -0.98, 0.98)
        barrier_shape = -eta / np.maximum(1.0 - eta * eta, 1e-3) ** 2
        barrier_velocity = float(args.joint_limit_barrier_gain) * barrier_shape * joint_half_range
        return center_velocity + np.clip(barrier_velocity, -float(args.max_nullspace_speed), float(args.max_nullspace_speed))

    def x_spin_ready_score(q_arm: np.ndarray) -> float:
        # Heuristic readiness score for future object-X spinning. Larger means the arm
        # is farther from joint limits and closer to the middle of its ranges.
        margin, cost = joint_centering_metrics(q_arm)
        return float(max(0.0, min(1.0, 2.0 * margin)) / (1.0 + cost))

    def orthonormal_basis_perp(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = normalize_vec(axis)
        ref = np.asarray([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(ref, z))) > 0.85:
            ref = np.asarray([0.0, 1.0, 0.0], dtype=float)
        b1 = normalize_vec(ref - float(np.dot(ref, z)) * z)
        b2 = normalize_vec(np.cross(z, b1))
        return b1, b2

    def signed_twist_about_axis(current_mat: np.ndarray, target_mat: np.ndarray, axis_world: np.ndarray) -> float:
        axis = normalize_vec(axis_world)
        x_cur = np.asarray(current_mat, dtype=float).reshape(3, 3)[:, 0]
        x_tar = np.asarray(target_mat, dtype=float).reshape(3, 3)[:, 0]
        x_cur = x_cur - float(np.dot(x_cur, axis)) * axis
        x_tar = x_tar - float(np.dot(x_tar, axis)) * axis
        if float(np.linalg.norm(x_cur)) < 1e-8 or float(np.linalg.norm(x_tar)) < 1e-8:
            return 0.0
        x_cur = normalize_vec(x_cur)
        x_tar = normalize_vec(x_tar)
        return float(math.atan2(float(np.dot(axis, np.cross(x_cur, x_tar))), float(np.dot(x_cur, x_tar))))

    def rot_z(angle_rad: float) -> np.ndarray:
        c = math.cos(float(angle_rad))
        s = math.sin(float(angle_rad))
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    def unwrap_angle_near(angle_rad: float, reference_rad: float) -> float:
        # Map an angle in [-pi, pi] to the equivalent unlimited-joint angle nearest
        # to the reference. This avoids wrist_spin jumping by 2*pi when the target
        # crosses the +/-180 degree display boundary.
        return float(reference_rad + math.atan2(math.sin(angle_rad - reference_rad), math.cos(angle_rad - reference_rad)))

    def compute_absolute_wrist_spin_target(
        site_mat_with_spin: np.ndarray,
        target_mat: np.ndarray,
        current_spin_rad: float,
    ) -> float:
        # 5D arm IK intentionally ignores the final twist about the tool/object Z axis.
        # To still make the full mocap-target frame match, remove the currently applied
        # wrist_spin from attachment_site and compute the absolute hinge angle that
        # rotates this zero-spin site frame into the target x/y phase.
        #
        # For the current XML the hinge axis, hand Z and attachment_site Z are collinear;
        # attachment_site's fixed local quat is also a pure Z-phase, so right-multiplying
        # by Rz(-current_spin) cleanly reconstructs the zero-spin site frame.
        site_no_spin_mat = np.asarray(site_mat_with_spin, dtype=float).reshape(3, 3) @ rot_z(-current_spin_rad)
        z_target = normalize_vec(np.asarray(target_mat, dtype=float).reshape(3, 3)[:, 2])
        desired_principal = signed_twist_about_axis(site_no_spin_mat, target_mat, z_target)
        return unwrap_angle_near(desired_principal, current_spin_rad)

    def move_toward(value: float, target: float, max_delta: float) -> float:
        return float(value + np.clip(target - value, -abs(max_delta), abs(max_delta)))

    if gripper_actuator_id is not None:
        data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED_CTRL
    mujoco.mj_forward(model, data)

    try:
        object_body_id: Optional[int] = int(model.body("attached_grasp_ellipsoid").id)
    except KeyError:
        object_body_id = None
    site_to_object_pos = OBJECT_OFFSET_IN_EE.copy()
    site_to_object_mat = np.eye(3, dtype=float)
    if object_body_id is not None:
        site_mat0 = np.asarray(data.site(site_id).xmat, dtype=float).reshape(3, 3)
        object_mat0 = np.asarray(data.body(object_body_id).xmat, dtype=float).reshape(3, 3)
        site_to_object_pos = site_mat0.T @ (np.asarray(data.body(object_body_id).xpos, dtype=float) - np.asarray(data.site(site_id).xpos, dtype=float))
        site_to_object_mat = site_mat0.T @ object_mat0

    def maybe_geom(name: str) -> Optional[int]:
        try:
            return int(model.geom(name).id)
        except KeyError:
            return None

    ee_geom_ids = [
        gid
        for gid in [maybe_geom("ee_axis_x"), maybe_geom("ee_axis_y"), maybe_geom("ee_axis_z")]
        if gid is not None
    ]
    ee_site_ids = [site_id]
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

    def compensation_vec() -> np.ndarray:
        return np.asarray([0.0, 0.0, float(shared.mocap_z_comp)], dtype=float)

    def compensated_target_pos() -> np.ndarray:
        return data.mocap_pos[mocap_id] + compensation_vec()

    def current_site_quat() -> np.ndarray:
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, data.site(site_id).xmat)
        return normalize_quat(quat)

    def current_pose() -> np.ndarray:
        return np.concatenate([np.array(data.site(site_id).xpos, copy=True), rpy_from_mat(data.site(site_id).xmat)])

    def current_pose_quat() -> np.ndarray:
        return np.concatenate([np.array(data.site(site_id).xpos, copy=True), current_site_quat()])

    def current_object_pose_quat() -> tuple[np.ndarray, np.ndarray]:
        if object_body_id is not None:
            object_pos = np.array(data.body(object_body_id).xpos, copy=True)
            object_quat = mat_to_quat(np.asarray(data.body(object_body_id).xmat, dtype=float).reshape(3, 3))
            return object_pos, object_quat
        site_quat = current_site_quat()
        site_mat = mat_from_quat(site_quat)
        object_pos = np.array(data.site(site_id).xpos, copy=True) + site_mat @ OBJECT_OFFSET_IN_EE
        object_quat = site_quat.copy()
        return object_pos, object_quat

    def current_object_pose_rpy() -> np.ndarray:
        object_pos, object_quat = current_object_pose_quat()
        return np.concatenate([object_pos, rpy_from_mat(mat_from_quat(object_quat))])

    def mat_to_quat(mat: np.ndarray) -> np.ndarray:
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).reshape(9))
        return normalize_quat(quat)

    def quat_from_z_axis_and_x_hint(z_axis_world: np.ndarray, x_hint_world: np.ndarray) -> np.ndarray:
        z_axis = normalize_vec(z_axis_world)
        x_hint = normalize_vec(x_hint_world)
        x_axis = x_hint - float(np.dot(x_hint, z_axis)) * z_axis
        if float(np.linalg.norm(x_axis)) < 1e-8:
            # Use a stable fallback that is not parallel to the requested z-axis.
            for fallback in (np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])):
                x_axis = fallback - float(np.dot(fallback, z_axis)) * z_axis
                if float(np.linalg.norm(x_axis)) >= 1e-8:
                    break
        x_axis = normalize_vec(x_axis)
        y_axis = normalize_vec(np.cross(z_axis, x_axis))
        x_axis = normalize_vec(np.cross(y_axis, z_axis))
        return mat_to_quat(np.column_stack([x_axis, y_axis, z_axis]))

    def pose_command_from_object_pose(object_position: np.ndarray, object_quat: np.ndarray) -> PoseCommand:
        # Invert the fixed transform from attachment_site S to the actual attached object O:
        #   T_WO = T_WS * T_SO  =>  T_WS = T_WO * inv(T_SO).
        # If the object body is unavailable, site_to_object_* falls back to the historical
        # +0.17m same-orientation approximation, preserving old behavior.
        object_quat = normalize_quat(object_quat)
        object_mat = mat_from_quat(object_quat)
        ee_mat = object_mat @ site_to_object_mat.T
        ee_position = np.asarray(object_position, dtype=float) - ee_mat @ site_to_object_pos
        return pose_command_from_position_quat(ee_position, mat_to_quat(ee_mat))

    def set_mocap_from_object_pose(object_position: np.ndarray, object_quat: np.ndarray) -> None:
        set_mocap_pose(pose_command_from_object_pose(object_position, object_quat))

    def set_mocap_pose(cmd: PoseCommand) -> None:
        data.mocap_pos[mocap_id] = cmd.position
        data.mocap_quat[mocap_id] = cmd.quat

    def snap_mocap_to_site() -> None:
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, data.site(site_id).xmat)
        data.mocap_pos[mocap_id] = data.site(site_id).xpos - compensation_vec()
        data.mocap_quat[mocap_id] = quat

    def orientation_error() -> float:
        site_quat = np.zeros(4, dtype=float)
        site_quat_conj = np.zeros(4, dtype=float)
        error_quat = np.zeros(4, dtype=float)
        omega = np.zeros(3, dtype=float)
        mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
        mujoco.mju_quat2Vel(omega, error_quat, 1.0)
        return float(np.linalg.norm(omega))

    def clip_joints(q: np.ndarray) -> None:
        for jid, qpos_id in zip(joint_ids, qpos_ids, strict=True):
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                q[qpos_id] = np.clip(q[qpos_id], lo, hi)

    def clip_arm_values(q_arm: np.ndarray, safety_margin: float = 0.02) -> np.ndarray:
        q_arm = np.asarray(q_arm, dtype=float).copy()
        for i, jid in enumerate(joint_ids):
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                span = max(float(hi - lo), 1e-6)
                margin = min(max(float(safety_margin), 0.0), 0.20 * span)
                q_arm[i] = float(np.clip(q_arm[i], float(lo) + margin, float(hi) - margin))
        return q_arm

    def set_arm_qpos_for_data(d: Any, q_arm: np.ndarray) -> None:
        d.qpos[qpos_ids] = clip_arm_values(q_arm, safety_margin=0.01)
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)

    def build_5d_error_and_jacobian_for_data(
        d: Any,
        target_pos: np.ndarray,
        target_mat: np.ndarray,
        velocity_scaled: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        site_pos = np.asarray(d.site(site_id).xpos, dtype=float)
        site_mat_local = np.asarray(d.site(site_id).xmat, dtype=float).reshape(3, 3)
        dx_local = np.asarray(target_pos, dtype=float) - site_pos
        z_cur_local = normalize_vec(site_mat_local[:, 2])
        z_target_local = normalize_vec(np.asarray(target_mat, dtype=float).reshape(3, 3)[:, 2])
        z_cross_local = np.cross(z_cur_local, z_target_local)
        z_cross_norm_local = float(np.linalg.norm(z_cross_local))
        z_axis_error_local = float(math.atan2(z_cross_norm_local, float(np.dot(z_cur_local, z_target_local))))
        z_axis_error_vec_local = (
            z_cross_local / z_cross_norm_local * z_axis_error_local
            if z_cross_norm_local > 1e-9
            else np.zeros(3, dtype=float)
        )

        b1_local, b2_local = orthonormal_basis_perp(z_cur_local)
        task = np.zeros(5, dtype=float)
        if velocity_scaled:
            task[:3] = float(args.kpos) * dx_local / float(args.dt)
            omega_z_local = float(args.kori) * z_axis_error_vec_local / float(args.dt)
        else:
            task[:3] = dx_local
            omega_z_local = z_axis_error_vec_local
        task[3:] = np.asarray([float(np.dot(b1_local, omega_z_local)), float(np.dot(b2_local, omega_z_local))], dtype=float)

        jac_tmp = np.zeros((6, model.nv), dtype=float)
        mujoco.mj_jacSite(model, d, jac_tmp[:3], jac_tmp[3:], site_id)
        jac_pos_local = jac_tmp[:3, dof_ids]
        jac_rot_local = jac_tmp[3:, dof_ids]
        jac_task = np.vstack([jac_pos_local, np.vstack([b1_local, b2_local]) @ jac_rot_local])
        return task, jac_task, float(np.linalg.norm(dx_local)), z_axis_error_local

    def dls_pinv_5d(jac_task: np.ndarray, damping_scale: float = 1.0) -> np.ndarray:
        lam = max(float(args.damping) * float(damping_scale), 1e-6)
        return jac_task.T @ np.linalg.solve(jac_task @ jac_task.T + (lam * lam) * np.eye(5), np.eye(5))

    def settle_arm_to_5d_target(
        start_q_arm: np.ndarray,
        target_pos: np.ndarray,
        target_mat: np.ndarray,
        attractor_q: Optional[np.ndarray] = None,
        iterations: int = 70,
    ) -> tuple[np.ndarray, float, float]:
        sim = mujoco.MjData(model)
        sim.qpos[:] = data.qpos
        sim.qvel[:] = 0.0
        set_arm_qpos_for_data(sim, start_q_arm)
        q_local = np.array(sim.qpos[qpos_ids], copy=True)
        for _ in range(max(1, int(iterations))):
            task_local, jac_local, pos_err_local, z_err_local = build_5d_error_and_jacobian_for_data(
                sim, target_pos, target_mat, velocity_scaled=False
            )
            if pos_err_local < 0.0015 and z_err_local < 0.015:
                break
            jsharp_local = dls_pinv_5d(jac_local, damping_scale=8.0)
            dq_local = jsharp_local @ task_local
            if attractor_q is not None:
                null_local = np.eye(len(dof_ids)) - jsharp_local @ jac_local
                dq_local += null_local @ (0.18 * (np.asarray(attractor_q, dtype=float) - q_local))
            step_max = float(np.max(np.abs(dq_local))) if len(dq_local) else 0.0
            if step_max > 0.10:
                dq_local *= 0.10 / step_max
            q_local = clip_arm_values(q_local + dq_local, safety_margin=0.015)
            set_arm_qpos_for_data(sim, q_local)
        _, _, pos_err_final, z_err_final = build_5d_error_and_jacobian_for_data(sim, target_pos, target_mat, velocity_scaled=False)
        return q_local, pos_err_final, z_err_final

    def estimate_x_spin_range_deg(
        start_q_arm: np.ndarray,
        object_position: np.ndarray,
        object_quat: np.ndarray,
        direction: float,
    ) -> float:
        max_scan = abs(float(args.x_spin_prepare_scan_deg))
        step_deg = max(3.0, abs(float(args.x_spin_prepare_scan_step_deg)))
        q_rollout = np.asarray(start_q_arm, dtype=float).copy()
        reached = 0.0
        angle = step_deg
        while angle <= max_scan + 1e-9:
            q_delta = quat_from_axis_angle([1.0, 0.0, 0.0], math.radians(direction * angle))
            object_target_quat = quat_mul(object_quat, q_delta)
            ee_cmd = pose_command_from_object_pose(object_position, object_target_quat)
            q_rollout, pos_err_roll, z_err_roll = settle_arm_to_5d_target(
                q_rollout, ee_cmd.position, mat_from_quat(ee_cmd.quat), attractor_q=q_rollout, iterations=18
            )
            margin_roll, _ = joint_centering_metrics(q_rollout)
            if pos_err_roll > 0.010 or z_err_roll > 0.10 or margin_roll < 0.018:
                break
            reached = angle
            angle += step_deg
        return float(reached)

    def nullspace_basis_for_current_task(target_pos: np.ndarray, target_mat: np.ndarray) -> np.ndarray:
        _, jac_task, _, _ = build_5d_error_and_jacobian_for_data(data, target_pos, target_mat, velocity_scaled=False)
        jsharp_current = dls_pinv_5d(jac_task, damping_scale=8.0)
        null_current = np.eye(len(dof_ids)) - jsharp_current @ jac_task
        u, svals, _ = np.linalg.svd(null_current)
        basis = u[:, svals > 0.35]
        if basis.shape[1] == 0:
            return null_current @ np.eye(len(dof_ids))[:, :1]
        return basis[:, : min(2, basis.shape[1])]

    def plan_x_spin_prepare_target() -> tuple[np.ndarray, str]:
        # This is intentionally stronger than the normal online nullspace centering.
        # It samples the current 5D nullspace, settles each candidate back onto the
        # current object/EE task, then explicitly rolls out +/- object-X spin to score
        # how much rotation remains feasible before joint limits or IK error appear.
        current_q = np.array(data.qpos[qpos_ids], copy=True)
        target_pos_now = compensated_target_pos().copy()
        target_mat_now = mat_from_quat(data.mocap_quat[mocap_id])
        object_pos_now, object_quat_now = current_object_pose_quat()
        basis = nullspace_basis_for_current_task(target_pos_now, target_mat_now)
        search_radius = max(0.05, float(args.x_spin_prepare_search_radius))
        max_candidates = max(5, int(args.x_spin_prepare_candidates))

        candidate_seeds: list[np.ndarray] = [current_q.copy()]
        center_projected = current_q.copy()
        try:
            _, jac_task_now, _, _ = build_5d_error_and_jacobian_for_data(data, target_pos_now, target_mat_now, velocity_scaled=False)
            jsharp_now = dls_pinv_5d(jac_task_now, damping_scale=8.0)
            null_now = np.eye(len(dof_ids)) - jsharp_now @ jac_task_now
            center_projected = current_q + null_now @ (joint_mid - current_q)
            candidate_seeds.append(center_projected)
        except np.linalg.LinAlgError:
            pass

        if basis.shape[1] == 1:
            for a in np.linspace(-search_radius, search_radius, max_candidates - len(candidate_seeds)):
                candidate_seeds.append(current_q + basis[:, 0] * float(a))
        else:
            levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
            for a in levels:
                for b in levels:
                    candidate_seeds.append(current_q + search_radius * (float(a) * basis[:, 0] + float(b) * basis[:, 1]))

        # Keep a compact deterministic list, with the center-projected seed near the front.
        compact: list[np.ndarray] = []
        for seed in candidate_seeds:
            seed = clip_arm_values(seed, safety_margin=0.02)
            if not any(float(np.linalg.norm(seed - prev)) < 0.03 for prev in compact):
                compact.append(seed)
            if len(compact) >= max_candidates:
                break

        best_q = current_q.copy()
        best_score = -1e9
        best_report = ""
        current_margin, current_cost = joint_centering_metrics(current_q)
        for index, seed in enumerate(compact):
            settled_q, pos_err_settle, z_err_settle = settle_arm_to_5d_target(
                seed, target_pos_now, target_mat_now, attractor_q=seed, iterations=90
            )
            if pos_err_settle > 0.006 or z_err_settle > 0.06:
                continue
            margin, cost = joint_centering_metrics(settled_q)
            plus_range = estimate_x_spin_range_deg(settled_q, object_pos_now, object_quat_now, +1.0)
            minus_range = estimate_x_spin_range_deg(settled_q, object_pos_now, object_quat_now, -1.0)
            scan_denom = max(1.0, 2.0 * abs(float(args.x_spin_prepare_scan_deg)))
            range_score = (plus_range + minus_range) / scan_denom
            ready = x_spin_ready_score(settled_q)
            motion_penalty = 0.05 * float(np.linalg.norm(settled_q - current_q))
            score = 4.0 * range_score + 1.5 * ready + 0.8 * margin - 0.4 * cost - motion_penalty
            if score > best_score:
                best_score = score
                best_q = settled_q.copy()
                best_report = (
                    f"candidate {index + 1}/{len(compact)}, +/-X range {minus_range:.0f}/{plus_range:.0f} deg, "
                    f"margin {margin:.3f}, center-cost {cost:.3f}"
                )

        if best_score < -1e8:
            # Fall back to a visible but still task-safe projected center move.
            fallback_q, pos_err_fb, z_err_fb = settle_arm_to_5d_target(
                center_projected, target_pos_now, target_mat_now, attractor_q=center_projected, iterations=100
            )
            best_q = fallback_q
            margin_fb, cost_fb = joint_centering_metrics(best_q)
            best_report = (
                f"fallback projected-center target, pos_err {pos_err_fb:.4f}, z_err {z_err_fb:.3f}, "
                f"margin {margin_fb:.3f}, center-cost {cost_fb:.3f}"
            )

        planned_motion = float(np.linalg.norm(best_q - current_q))
        if planned_motion < 0.04:
            best_report += " | 当前构型已经接近局部最优，计划移动很小"
        else:
            best_report += f" | planned joint motion {planned_motion:.3f} rad"
        best_report += f" | before margin {current_margin:.3f}, center-cost {current_cost:.3f}"
        return best_q, best_report

    snap_mocap_to_site()

    print("[3/5] Opening MuJoCo viewer.", flush=True)
    print(
        f"[speed] kpos={args.kpos:.3f}, kori={args.kori:.3f}, max_angvel={args.max_angvel:.3f} rad/s",
        flush=True,
    )
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=args.show_mujoco_ui,
        show_right_ui=args.show_mujoco_ui,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE if args.show_site_frame_glyphs else mujoco.mjtFrame.mjFRAME_NONE
        print("[4/5] Controller running. Drag the mocap target or use Move One Step.", flush=True)

        jac_full = np.zeros((6, model.nv), dtype=float)
        diag = float(args.damping) * np.eye(6)
        eye_arm = np.eye(len(dof_ids))
        twist = np.zeros(6, dtype=float)
        site_quat = np.zeros(4, dtype=float)
        site_quat_conj = np.zeros(4, dtype=float)
        error_quat = np.zeros(4, dtype=float)
        motion = MotionState(remaining_goals=[])
        arm_posture_reference = q0_arm.copy()
        joint_centering_enabled = not bool(args.disable_joint_centering)
        joint_centering_active = joint_centering_enabled
        running = True
        status = "ready"
        last_state_time = 0.0
        x_spin_prepare_until = 0.0
        forced_nullspace_reason = ""
        x_spin_prepare_q_target: Optional[np.ndarray] = None
        x_spin_prepare_plan_report = ""
        wrist_spin_target_rad = float(data.qpos[wrist_spin_qpos_id]) if wrist_spin_qpos_id is not None else 0.0

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
                        waypoints = [pose_array_to_command(item) for item in command.get("waypoints", [])]
                        if waypoints:
                            motion.active_mode = str(command.get("mode", "move_one_step"))
                            motion.current_goal = waypoints[0]
                            motion.remaining_goals = waypoints[1:]
                            motion.active_goal_index = 1
                            motion.total_goal_count = len(waypoints)
                            set_mocap_pose(motion.current_goal)
                            status = f"start {motion.active_mode}: waypoint {motion.active_goal_index}/{motion.total_goal_count}"
                    elif ctype == "object_move":
                        object_cmd = pose_array_to_command(command.get("waypoint", []))
                        ee_cmd = pose_command_from_object_pose(object_cmd.position, object_cmd.quat)
                        motion.active_mode = "object_move_one_step"
                        joint_centering_active = joint_centering_enabled
                        motion.current_goal = ee_cmd
                        motion.remaining_goals = []
                        motion.active_goal_index = 1
                        motion.total_goal_count = 1
                        set_mocap_pose(ee_cmd)
                        status = "start object_move_one_step: object pose -> EE target"
                    elif ctype == "object_preset":
                        object_pos, object_quat = current_object_pose_quat()
                        object_mat = mat_from_quat(object_quat)
                        z_target = world_direction_from_name(str(command.get("direction", "")))
                        x_hint = object_mat[:, 0]
                        object_target_quat = quat_from_z_axis_and_x_hint(z_target, x_hint)
                        ee_cmd = pose_command_from_object_pose(object_pos, object_target_quat)
                        motion.active_mode = f"object_preset_{command.get('direction', '')}"
                        joint_centering_active = joint_centering_enabled
                        motion.current_goal = ee_cmd
                        motion.remaining_goals = []
                        motion.active_goal_index = 1
                        motion.total_goal_count = 1
                        set_mocap_pose(ee_cmd)
                        status = f"object preset: z-axis -> {command.get('direction', '')}"
                    elif ctype == "object_rotate_body":
                        axis_name = str(command.get("axis", "")).lower()
                        angle_deg = float(command.get("angle_deg", 0.0))
                        if axis_name not in {"x", "y", "z"}:
                            raise ValueError("object_rotate_body axis must be x/y/z")
                        object_pos, object_quat = current_object_pose_quat()
                        axis_vec = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis_name]
                        q_delta = quat_from_axis_angle(axis_vec, math.radians(angle_deg))
                        # Right multiplication: rotate around the object's own current body axis.
                        object_target_quat = quat_mul(object_quat, q_delta)
                        ee_cmd = pose_command_from_object_pose(object_pos, object_target_quat)
                        motion.active_mode = f"object_rot_{axis_name}"
                        joint_centering_active = joint_centering_enabled
                        motion.current_goal = ee_cmd
                        motion.remaining_goals = []
                        motion.active_goal_index = 1
                        motion.total_goal_count = 1
                        set_mocap_pose(ee_cmd)
                        status = f"object rotate around own {axis_name.upper()}: {angle_deg:.3f} deg"
                    elif ctype == "object_prepare_x_spin":
                        # Keep the current full object/EE pose as the target. Unlike the old
                        # version, this first plans a concrete redundant joint target by testing
                        # candidate nullspace configurations for future +/- object-X spin range.
                        snap_mocap_to_site()
                        if wrist_spin_qpos_id is not None:
                            wrist_spin_target_rad = float(data.qpos[wrist_spin_qpos_id])
                        motion.clear()
                        motion.active_mode = "x_spin_prepare"
                        seconds = command.get("seconds", None)
                        duration = float(args.x_spin_prepare_seconds if seconds is None else seconds)
                        x_spin_prepare_q_target, x_spin_prepare_plan_report = plan_x_spin_prepare_target()
                        x_spin_prepare_until = time.time() + max(0.0, duration)
                        forced_nullspace_reason = "x_spin_prepare"
                        joint_centering_active = joint_centering_enabled
                        status = f"x-spin prepare planned for {duration:.2f}s: {x_spin_prepare_plan_report}"
                    elif ctype == "stop":
                        motion.clear()
                        x_spin_prepare_until = 0.0
                        x_spin_prepare_q_target = None
                        x_spin_prepare_plan_report = ""
                        forced_nullspace_reason = ""
                        joint_centering_active = joint_centering_enabled
                        arm_posture_reference = np.array(data.qpos[qpos_ids], copy=True)
                        snap_mocap_to_site()
                        if wrist_spin_qpos_id is not None:
                            wrist_spin_target_rad = float(data.qpos[wrist_spin_qpos_id])
                        status = "stopped; target snapped to current EE pose"
                    elif ctype == "ee_axes":
                        shared.show_ee_axes = bool(command.get("visible", True))
                    elif ctype == "mocap":
                        shared.show_mocap = bool(command.get("visible", True))
                    elif ctype == "offset":
                        shared.mocap_z_comp = float(command.get("mocap_z_comp", shared.mocap_z_comp))
                        snap_mocap_to_site()
                        if wrist_spin_qpos_id is not None:
                            wrist_spin_target_rad = float(data.qpos[wrist_spin_qpos_id])
                        status = f"mocap z compensation = {shared.mocap_z_comp:.4f} m"
                    elif ctype == "quit":
                        running = False
                except Exception as exc:
                    status = f"command failed: {exc}"

            apply_visibility(ee_geom_ids, ee_site_ids, bool(shared.show_ee_axes))
            apply_visibility(mocap_geom_ids, mocap_site_ids, bool(shared.show_mocap))

            if gripper_actuator_id is not None:
                data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED_CTRL

            if motion.current_goal is not None:
                pos_err_goal = float(np.linalg.norm(compensated_target_pos() - data.site(site_id).xpos))
                ori_err_goal = orientation_error()
                if pos_err_goal < float(args.position_tolerance) and ori_err_goal < float(args.orientation_tolerance):
                    if motion.remaining_goals:
                        motion.current_goal = motion.remaining_goals.pop(0)
                        motion.active_goal_index += 1
                        set_mocap_pose(motion.current_goal)
                        status = f"continue waypoint {motion.active_goal_index}/{motion.total_goal_count}"
                    else:
                        status = f"{motion.active_mode} complete"
                        motion.clear()
                        joint_centering_active = joint_centering_enabled

            dx = compensated_target_pos() - data.site(site_id).xpos
            target_mat = mat_from_quat(data.mocap_quat[mocap_id])
            site_mat = np.asarray(data.site(site_id).xmat, dtype=float).reshape(3, 3)
            z_cur = normalize_vec(site_mat[:, 2])
            z_target = normalize_vec(target_mat[:, 2])
            z_cross = np.cross(z_cur, z_target)
            z_cross_norm = float(np.linalg.norm(z_cross))
            z_axis_error = float(math.atan2(z_cross_norm, float(np.dot(z_cur, z_target))))
            z_axis_error_vec = (z_cross / z_cross_norm * z_axis_error) if z_cross_norm > 1e-9 else np.zeros(3, dtype=float)

            # 5D task: 3D position + 2D z-axis direction. The final twist about z is
            # removed from the FR3 IK and handled only by wrist_spin below.
            b1, b2 = orthonormal_basis_perp(z_cur)
            task5 = np.zeros(5, dtype=float)
            task5[:3] = float(args.kpos) * dx / float(args.dt)
            omega_z = float(args.kori) * z_axis_error_vec / float(args.dt)
            task5[3:] = np.asarray([float(np.dot(b1, omega_z)), float(np.dot(b2, omega_z))], dtype=float)

            mujoco.mj_jacSite(model, data, jac_full[:3], jac_full[3:], site_id)
            jac_pos = jac_full[:3, dof_ids]
            jac_rot = jac_full[3:, dof_ids]
            jac5 = np.vstack([jac_pos, np.vstack([b1, b2]) @ jac_rot])
            diag5 = float(args.damping) * np.eye(5)
            q_arm = np.array(data.qpos[qpos_ids], copy=True)
            primary_scale = (float(np.linalg.norm(dx)) / max(float(args.position_tolerance), 1e-6)) + (z_axis_error / max(float(args.orientation_tolerance), 1e-6))
            pure_twist_or_steady = primary_scale < float(args.joint_secondary_gate)
            force_nullspace = bool(x_spin_prepare_until > time.time())
            if force_nullspace and x_spin_prepare_q_target is not None:
                target_distance = float(np.linalg.norm(x_spin_prepare_q_target - q_arm))
                if target_distance < 0.025:
                    x_spin_prepare_until = 0.0
                    force_nullspace = False
                    forced_nullspace_reason = ""
                    x_spin_prepare_q_target = None
                    if motion.active_mode == "x_spin_prepare":
                        motion.clear()
                    status = f"x-spin prepare reached planned joint target: {x_spin_prepare_plan_report}"
            if x_spin_prepare_until > 0.0 and not force_nullspace:
                forced_nullspace_reason = ""
                x_spin_prepare_q_target = None
                if motion.active_mode == "x_spin_prepare":
                    motion.clear()
                    status = f"x-spin prepare complete: {x_spin_prepare_plan_report}"
            try:
                jac5_dls = jac5.T @ np.linalg.solve(jac5 @ jac5.T + diag5, np.eye(5))
                dq_arm = jac5_dls @ task5
                nullspace_projector = eye_arm - jac5_dls @ jac5
                if joint_centering_active:
                    if force_nullspace:
                        gamma_secondary = 1.0
                    elif pure_twist_or_steady:
                        gamma_secondary = 0.0
                    else:
                        gamma_secondary = float(np.clip(primary_scale / max(float(args.joint_secondary_gate), 1e-6), 0.0, 1.0))
                    if force_nullspace and x_spin_prepare_q_target is not None:
                        # During explicit X-spin preparation, move toward the planned
                        # nullspace target. The primary 5D term still protects xyz + z-axis.
                        dq_null = (
                            float(args.x_spin_prepare_gain) * (x_spin_prepare_q_target - q_arm)
                            + 0.25 * joint_limit_avoidance_velocity(q_arm)
                        )
                    else:
                        dq_null = joint_limit_avoidance_velocity(q_arm)
                else:
                    gamma_secondary = 1.0
                    dq_null = float(args.kn) * (arm_posture_reference - q_arm)
                dq_arm += gamma_secondary * (nullspace_projector @ dq_null)
            except np.linalg.LinAlgError:
                dq_arm = np.zeros(len(dof_ids), dtype=float)
                status = "warning: 5D Jacobian solve failed"

            # wrist_spin closes the one remaining twist DOF. This must also run for
            # manual mocap-target rotations: the FR3 arm solves only xyz + z-axis
            # direction, while wrist_spin receives the absolute hinge target needed to
            # match the mocap target's x/y phase around that z axis.
            if wrist_spin_qpos_id is not None and wrist_spin_actuator_id is not None:
                current_spin = float(data.qpos[wrist_spin_qpos_id])
                desired_spin = compute_absolute_wrist_spin_target(site_mat, target_mat, current_spin)
                # Blend the absolute target with a gain, then rate-limit the actuator
                # target. Keeping wrist_spin_target_rad as a persistent setpoint makes
                # mocap dragging behave like a real closed-loop spin controller instead
                # of a weak per-frame qpos increment.
                desired_spin = current_spin + float(args.wrist_spin_gain) * (desired_spin - current_spin)
                max_step = math.radians(float(args.wrist_spin_speed_deg_s)) * float(args.dt)
                wrist_spin_target_rad = move_toward(wrist_spin_target_rad, desired_spin, max_step)
                data.ctrl[wrist_spin_actuator_id] = float(wrist_spin_target_rad)

            dq_abs_max = float(np.max(np.abs(dq_arm))) if len(dq_arm) else 0.0
            if dq_abs_max > float(args.max_angvel):
                dq_arm *= float(args.max_angvel) / dq_abs_max

            q = data.qpos.copy()
            dq_full = np.zeros(model.nv, dtype=float)
            dq_full[dof_ids] = dq_arm
            mujoco.mj_integratePos(model, q, dq_full, float(args.dt))
            clip_joints(q)
            data.ctrl[actuator_ids] = q[qpos_ids]
            mujoco.mj_step(model, data)

            viewer.sync()

            now = time.time()
            if now - last_state_time > 0.05:
                pose_rad = current_pose()
                pose_deg = np.concatenate([pose_rad[:3], np.rad2deg(pose_rad[3:])])
                pose_quat = current_pose_quat()[3:]
                object_pose_rad = current_object_pose_rpy()
                object_pose_deg = np.concatenate([object_pose_rad[:3], np.rad2deg(object_pose_rad[3:])])
                object_pos, object_quat = current_object_pose_quat()
                q_arm_state = np.array(data.qpos[qpos_ids], copy=True)
                joint_margin, joint_center_cost = joint_centering_metrics(q_arm_state)
                shared.update(
                    pose_rad=[float(x) for x in pose_rad],
                    pose_deg=[float(x) for x in pose_deg],
                    pose_quat=[float(x) for x in pose_quat],
                    object_pose_rad=[float(x) for x in object_pose_rad],
                    object_pose_deg=[float(x) for x in object_pose_deg],
                    object_pose_quat=[float(x) for x in object_quat],
                    joint_angles_rad=[float(x) for x in q_arm_state],
                    joint_angles_deg=[float(x) for x in np.rad2deg(q_arm_state)],
                    joint_limit_margin=float(joint_margin),
                    joint_center_cost=float(joint_center_cost),
                    joint_centering_active=bool(joint_centering_active),
                    x_spin_ready_score=float(x_spin_ready_score(q_arm_state)),
                    pos_err=float(np.linalg.norm(compensated_target_pos() - data.site(site_id).xpos)),
                    ori_err=orientation_error(),
                    status=status,
                    viewer_running=True,
                    waypoint_index=int(motion.active_goal_index),
                    waypoint_count=int(motion.total_goal_count),
                    active_mode=motion.active_mode or "idle",
                    show_ee_axes=bool(shared.show_ee_axes),
                    show_mocap=bool(shared.show_mocap),
                    mocap_z_comp=float(shared.mocap_z_comp),
                )
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
    parser = argparse.ArgumentParser(description="Minimal FR3 mocap target follower with Move One Step UI")
    parser.add_argument("--ui-client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ui-python", type=str, default=None)
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--no-terminal-commands", action="store_true")
    parser.add_argument("--xml", type=str, default=str(Path(__file__).with_name("base_develop.xml")))
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--kpos", type=float, default=3.6)
    parser.add_argument("--kori", type=float, default=3.6)
    parser.add_argument("--kn", type=float, default=5.0)
    parser.add_argument("--joint-centering-gain", type=float, default=2.5, help="Global nullspace gain for joint-centering optimization used by all motion modes")
    parser.add_argument("--joint-limit-barrier-gain", type=float, default=0.04, help="Extra nullspace repulsion gain near joint limits")
    parser.add_argument("--max-nullspace-speed", type=float, default=4.0, help="Clamp for joint-limit barrier secondary velocity, rad/s")
    parser.add_argument("--joint-secondary-gate", type=float, default=0.25, help="Normalized 5D primary-error gate below which secondary optimization is frozen unless explicitly forced")
    parser.add_argument("--x-spin-prepare-seconds", type=float, default=4.0, help="Duration of planned nullspace preparation before large object-X spin, seconds")
    parser.add_argument("--x-spin-prepare-gain", type=float, default=8.0, help="Nullspace gain used to move toward the planned X-spin preparation joint target")
    parser.add_argument("--x-spin-prepare-search-radius", type=float, default=1.1, help="Search radius in nullspace coordinates for X-spin preparation candidates, rad")
    parser.add_argument("--x-spin-prepare-scan-deg", type=float, default=180.0, help="Maximum +/- object-X spin angle tested by the preparation planner")
    parser.add_argument("--x-spin-prepare-scan-step-deg", type=float, default=30.0, help="Object-X spin scan step used by the preparation planner")
    parser.add_argument("--x-spin-prepare-candidates", type=int, default=13, help="Maximum number of nullspace candidates evaluated by the X-spin preparation planner")
    parser.add_argument("--wrist-spin-speed-deg-s", type=float, default=120.0, help="Maximum wrist_spin tracking speed in deg/s")
    parser.add_argument("--wrist-spin-gain", type=float, default=1.0, help="Proportional gain for wrist_spin twist closure")
    parser.add_argument("--disable-joint-centering", action="store_true", help="Disable global nullspace joint-centering optimization")
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--max-angvel", type=float, default=21.0)
    parser.add_argument("--position-tolerance", type=float, default=0.003)
    parser.add_argument("--orientation-tolerance", type=float, default=0.03)
    parser.add_argument("--hide-ee-axes", action="store_true")
    parser.add_argument("--hide-mocap", action="store_true")
    parser.add_argument("--mocap-z-comp", type=float, default=0.0)
    parser.add_argument("--real-gravity", action="store_true", help="Use MuJoCo's normal gravity instead of zero gravity")
    parser.add_argument("--gravity-comp", action="store_true", help="Enable robot body gravity compensation")
    parser.add_argument("--no-gravity-comp", action="store_false", dest="gravity_comp", help=argparse.SUPPRESS)
    parser.add_argument("--show-mujoco-ui", action="store_true")
    parser.add_argument("--show-site-frame-glyphs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ui_client:
        return run_ui_client(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
