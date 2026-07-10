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
from typing import Any, Callable, Optional

import numpy as np


SITE_NAME = "attachment_site"
MOCAP_NAME = "target"
HAND_BODY_NAME = "hand"
JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
ACTUATOR_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
GRIPPER_ACTUATOR_NAME = "gripper"
WRIST_SPIN_JOINT_NAME = "wrist_spin"
WRIST_SPIN_ACTUATOR_NAME = "wrist_spin"
GRIPPER_OPEN_CTRL = 255.0
# The gripper actuator settles at fingertip position = ctrl * GRIPPER_CTRL_TO_WIDTH
# per finger (tendon equilibrium of the affine bias actuator).
GRIPPER_CTRL_PER_FINGER_M = 0.01568627451 / 100.0
# Anti-windup band for the persistent arm control setpoint: the commanded joint
# targets may never leave this distance (rad) from the measured joint angles.
SETPOINT_TRACKING_BAND = 0.05

# ---------------------------------------------------------------------------
# Flexible pick-place knowledge base.
#
# Flexibility principle: grasp knowledge is attached to the component TYPE, not
# to taught waypoints. Instance poses are perceived at task start, so components
# may lie anywhere in reach with any yaw and the same spec still applies.
# ---------------------------------------------------------------------------
APPROACH_CLEARANCE = 0.10   # hover height above grasp/place poses, m
LIFT_HEIGHT = 0.12          # vertical retreat after grasping, m
PLACE_DROP = 0.002          # release the part this far above its seated pose, m
GRASP_FINGER_MARGIN = 0.0015  # visual finger closing margin per side, m


@dataclass(frozen=True)
class ComponentSpec:
    """Type-level grasp knowledge for one electrical component family."""

    type_name: str
    grip_width: float            # component width along the gripper closing axis, m
    half_height: float           # component center to top face, m
    grasp_depth: float           # how deep below the top face the EE site aims, m
    grasp_yaw_in_object: float   # EE x-axis yaw relative to the object x-axis, rad
    yaw_symmetry_rad: float      # placement yaw symmetry (0 = any yaw is fine)
    place_drop: float = PLACE_DROP
    seat_press_m: float = 0.0    # extra downward travel while still attached, before release
    release_dwell_s: float = 0.5


COMPONENT_SPECS: dict[str, ComponentSpec] = {
    "relay": ComponentSpec(
        type_name="relay",
        grip_width=0.024,
        half_height=0.020,
        grasp_depth=0.012,
        grasp_yaw_in_object=math.pi / 2.0,
        yaw_symmetry_rad=math.pi,
    ),
    "terminal": ComponentSpec(
        type_name="terminal",
        grip_width=0.016,
        half_height=0.010,
        grasp_depth=0.008,
        grasp_yaw_in_object=math.pi / 2.0,
        yaw_symmetry_rad=math.pi,
    ),
    "button": ComponentSpec(
        type_name="button",
        grip_width=0.020,
        half_height=0.011,
        grasp_depth=0.010,
        grasp_yaw_in_object=0.0,
        yaw_symmetry_rad=math.pi,
        place_drop=0.0,
        seat_press_m=0.004,
        release_dwell_s=1.2,
    ),
}

# Instance registry: which bodies in the scene belong to which component type.
COMPONENT_INSTANCES: dict[str, str] = {
    "relay_1": "relay",
    "relay_2": "relay",
    "terminal_1": "terminal",
    "button_1": "button",
}

# Demo orders in the project-plan JSON dialect. "instance" may be omitted, in
# which case the executor picks a free instance of the requested type.
ORDER_PRESETS: dict[str, list[dict[str, str]]] = {
    "A": [
        {"type": "relay", "instance": "relay_1", "target_slot": "slot_1"},
        {"type": "terminal", "instance": "terminal_1", "target_slot": "slot_5"},
    ],
    "B": [
        {"type": "relay", "instance": "relay_2", "target_slot": "slot_3"},
        {"type": "button", "instance": "button_1", "target_slot": "slot_2"},
        {"type": "terminal", "instance": "terminal_1", "target_slot": "slot_6"},
    ],
    # Full 4-piece assembly: all component instances, slot_4 (first use), both
    # 90-deg rotated slots (slot_3 / slot_6), and the button seat-press path.
    "C": [
        {"type": "relay", "instance": "relay_2", "target_slot": "slot_1"},
        {"type": "button", "instance": "button_1", "target_slot": "slot_4"},
        {"type": "relay", "instance": "relay_1", "target_slot": "slot_3"},
        {"type": "terminal", "instance": "terminal_1", "target_slot": "slot_6"},
    ],
}


@dataclass(frozen=True)
class PickPlaceTask:
    component: str
    slot: str


def gripper_close_ctrl_for_width(grip_width: float) -> float:
    """Gripper ctrl value whose finger equilibrium visually pinches the part."""
    finger_target = max(grip_width / 2.0 - GRASP_FINGER_MARGIN, 0.0)
    return float(np.clip(finger_target / GRIPPER_CTRL_PER_FINGER_M, 0.0, 255.0))


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


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    import mujoco

    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).reshape(9))
    return normalize_quat(quat)


def yaw_from_mat(mat: np.ndarray) -> float:
    r = np.asarray(mat, dtype=float).reshape(3, 3)
    return float(math.atan2(r[1, 0], r[0, 0]))


def rot_z_mat(angle_rad: float) -> np.ndarray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def top_down_ee_quat(world_yaw: float) -> np.ndarray:
    # Roll 180 deg puts the EE blue axis straight down; yaw sets the world-frame
    # heading of the EE x-axis, which is the finger closing direction.
    return quat_from_rpy(math.pi, 0.0, world_yaw)


def compute_grasp_ee_pose(
    object_pos: np.ndarray,
    object_yaw: float,
    spec: ComponentSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-down grasp EE (site) pose from the perceived object pose."""
    ee_yaw = float(object_yaw) + float(spec.grasp_yaw_in_object)
    grasp_z = float(object_pos[2]) + float(spec.half_height) - float(spec.grasp_depth)
    pos = np.asarray([float(object_pos[0]), float(object_pos[1]), grasp_z], dtype=float)
    return pos, top_down_ee_quat(ee_yaw)


def compute_place_ee_pose(
    object_target_pos: np.ndarray,
    object_target_yaw: float,
    attach_pos_site_obj: np.ndarray,
    attach_mat_site_obj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the recorded grasp transform: T_WS = T_WO_target * inv(T_SO).

    attach_pos/mat_site_obj is the object pose in the EE-site frame captured at
    the instant the weld engaged, so the placement inherits the exact measured
    grasp instead of the nominal design values.
    """
    object_mat = rot_z_mat(object_target_yaw)
    ee_mat = object_mat @ np.asarray(attach_mat_site_obj, dtype=float).T
    ee_pos = np.asarray(object_target_pos, dtype=float) - ee_mat @ np.asarray(attach_pos_site_obj, dtype=float)
    return ee_pos, mat_to_quat(ee_mat)


@dataclass
class SkillStep:
    """One leg of the pick-place skill: a pose goal plus an on-arrival action."""

    label: str
    pose: Optional[PoseCommand] = None
    pose_factory: Optional[Callable[[], "PoseCommand"]] = None
    action: str = ""          # "" | "grasp" | "release"
    dwell_s: float = 0.1
    pos_tol: float = 0.004
    ori_tol: float = 0.05

    def resolve_pose(self) -> "PoseCommand":
        if self.pose is not None:
            return self.pose
        if self.pose_factory is not None:
            return self.pose_factory()
        raise ValueError(f"skill step {self.label} has no pose")


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


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self.pose_rad = [0.0] * 6
        self.pose_deg = [0.0] * 6
        self.pose_quat = [1.0, 0.0, 0.0, 0.0]
        self.joint_angles_rad = [0.0] * 7
        self.joint_angles_deg = [0.0] * 7
        self.joint_limit_margin = 0.0
        self.joint_center_cost = 0.0
        self.joint_centering_active = True
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
        self.skill_task = ""
        self.skill_step_label = ""
        self.skill_step_index = 0
        self.skill_step_count = 0
        self.task_queue_len = 0
        self.held_component = ""
        self.available_components = list(COMPONENT_INSTANCES.keys())
        self.available_slots: list[str] = []
        self.server_time = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "pose_rad": list(self.pose_rad),
                "pose_deg": list(self.pose_deg),
                "pose_quat": list(self.pose_quat),
                "joint_angles_rad": list(self.joint_angles_rad),
                "joint_angles_deg": list(self.joint_angles_deg),
                "joint_limit_margin": float(self.joint_limit_margin),
                "joint_center_cost": float(self.joint_center_cost),
                "joint_centering_active": bool(self.joint_centering_active),
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
                "skill_task": str(self.skill_task),
                "skill_step_label": str(self.skill_step_label),
                "skill_step_index": int(self.skill_step_index),
                "skill_step_count": int(self.skill_step_count),
                "task_queue_len": int(self.task_queue_len),
                "held_component": str(self.held_component),
                "available_components": list(self.available_components),
                "available_slots": list(self.available_slots),
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
            elif self.path == "/pick_place":
                component = str(payload.get("component", ""))
                slot = str(payload.get("slot", ""))
                if not component or not slot:
                    raise ValueError("pick_place requires component and slot")
                self.shared.commands.put({"type": "pick_place", "component": component, "slot": slot})
                self._send_json({"ok": True, "component": component, "slot": slot})
            elif self.path == "/order":
                components = payload.get("components", None)
                preset = str(payload.get("preset", ""))
                if components is None and preset:
                    components = ORDER_PRESETS.get(preset.upper())
                    if components is None:
                        raise ValueError(f"unknown order preset: {preset}")
                if not isinstance(components, list) or not components:
                    raise ValueError("order requires components: [{type, instance?, target_slot}, ...]")
                self.shared.commands.put({"type": "order", "components": components, "order_id": str(payload.get("order_id", preset or "custom"))})
                self._send_json({"ok": True, "count": len(components)})
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
            self.setWindowTitle("flexible_pick_place - FR3 柔性取放")
            self.combos_filled = False
            self.last_pose_rad = [0.0] * 6
            self.last_pose_deg = [0.0] * 6
            self.last_pose_quat = [1.0, 0.0, 0.0, 0.0]

            layout = QVBoxLayout(self)

            state_group = QGroupBox("Current End Effector Pose")
            state_layout = QVBoxLayout(state_group)
            self.pose_rad_label = QLabel("waiting...")
            self.pose_deg_label = QLabel("waiting...")
            self.pose_quat_label = QLabel("quat: waiting...")
            self.joint_deg_label = QLabel("Joints deg: waiting...")
            self.joint_rad_label = QLabel("Joints rad: waiting...")
            self.joint_margin_label = QLabel("joint margin: waiting...")
            self.err_label = QLabel("pos_err: --   ori_err: --")
            self.status_label = QLabel("starting...")
            for label in (
                self.pose_rad_label,
                self.pose_deg_label,
                self.pose_quat_label,
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

            skill_group = QGroupBox("Flexible Pick and Place / 柔性取放")
            skill_layout = QVBoxLayout(skill_group)

            order_row = QHBoxLayout()
            order_a_btn = QPushButton("执行订单 A (2件)")
            order_a_btn.clicked.connect(lambda: self.post("/order", {"preset": "A"}))
            order_b_btn = QPushButton("执行订单 B (3件)")
            order_b_btn.clicked.connect(lambda: self.post("/order", {"preset": "B"}))
            order_c_btn = QPushButton("执行订单 C (4件)")
            order_c_btn.clicked.connect(lambda: self.post("/order", {"preset": "C"}))
            order_stop_btn = QPushButton("停止任务")
            order_stop_btn.clicked.connect(lambda: self.post("/stop", {}))
            order_row.addWidget(order_a_btn)
            order_row.addWidget(order_b_btn)
            order_row.addWidget(order_c_btn)
            order_row.addWidget(order_stop_btn)
            order_row.addStretch(1)
            skill_layout.addLayout(order_row)

            single_row = QHBoxLayout()
            single_row.addWidget(QLabel("元件:"))
            self.component_combo = QComboBox()
            single_row.addWidget(self.component_combo)
            single_row.addWidget(QLabel("目标槽位:"))
            self.slot_combo = QComboBox()
            single_row.addWidget(self.slot_combo)
            single_btn = QPushButton("单件取放")
            single_btn.clicked.connect(self.run_single_pick_place)
            single_row.addWidget(single_btn)
            single_row.addStretch(1)
            skill_layout.addLayout(single_row)

            self.skill_status_label = QLabel("skill: idle")
            self.make_label_copyable(self.skill_status_label)
            skill_layout.addWidget(self.skill_status_label)
            layout.addWidget(skill_group)

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

        def run_single_pick_place(self) -> None:
            component = self.component_combo.currentText()
            slot = self.slot_combo.currentText()
            if not component or not slot:
                QMessageBox.warning(self, "取放任务", "元件/槽位列表尚未加载")
                return
            self.post("/pick_place", {"component": component, "slot": slot})

        def refresh_state(self) -> None:
            try:
                state = get_json(base_url + "/state", timeout=0.5)
                self.last_pose_rad = [float(x) for x in state.get("pose_rad", [0.0] * 6)]
                self.last_pose_deg = [float(x) for x in state.get("pose_deg", [0.0] * 6)]
                self.last_pose_quat = [float(x) for x in state.get("pose_quat", [1.0, 0.0, 0.0, 0.0])]
                self.pose_rad_label.setText("EE rad: " + format_pose(self.last_pose_rad))
                self.pose_deg_label.setText("EE deg: " + format_pose(self.last_pose_deg))
                self.pose_quat_label.setText("EE quat [qw qx qy qz]: " + format_pose(self.last_pose_quat))
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

                if not self.combos_filled:
                    components = [str(x) for x in state.get("available_components", [])]
                    slots = [str(x) for x in state.get("available_slots", [])]
                    if components and slots:
                        self.component_combo.addItems(components)
                        self.slot_combo.addItems(slots)
                        self.combos_filled = True
                task = str(state.get("skill_task", ""))
                step_label = str(state.get("skill_step_label", ""))
                step_i = int(state.get("skill_step_index", 0))
                step_n = int(state.get("skill_step_count", 0))
                queue_len = int(state.get("task_queue_len", 0))
                held = str(state.get("held_component", ""))
                if task:
                    self.skill_status_label.setText(
                        f"skill: {task}  step {step_i}/{step_n} [{step_label}]  queue: {queue_len}  held: {held or '-'}"
                    )
                else:
                    self.skill_status_label.setText(f"skill: idle  queue: {queue_len}  held: {held or '-'}")

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
    print("[UI] flexible_pick_place PySide6 panel opened.", flush=True)
    return int(app.exec())


def print_terminal_help() -> None:
    print(
        "\n[终端命令]\n"
        "  pick <元件> <槽位>   单件柔性取放，例如: pick relay_1 slot_1\n"
        "  order_a/b/c         执行预设订单 A / B / C\n"
        "  ee_axes on/off      显示/隐藏机械臂末端坐标轴\n"
        "  mocap on/off        显示/隐藏 mocap target\n"
        "  offset 0.0          设置 mocap z 补偿\n"
        "  stop                停止当前任务/Move One Step\n"
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
            elif cmd.startswith("pick "):
                parts = cmd.split()
                if len(parts) != 3:
                    print("[terminal] 用法: pick <元件> <槽位>", flush=True)
                    continue
                shared.commands.put({"type": "pick_place", "component": parts[1], "slot": parts[2]})
            elif cmd.startswith("order_"):
                preset = cmd.split("_", 1)[1].upper()
                if preset not in ORDER_PRESETS:
                    print(f"[terminal] 未知订单预设: {preset}", flush=True)
                    continue
                shared.commands.put({"type": "order", "components": ORDER_PRESETS[preset], "order_id": preset})
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
    if args.zero_gravity:
        model.opt.gravity[:] = 0.0

    site_id = int(model.site(SITE_NAME).id)
    mocap_id = int(model.body(MOCAP_NAME).mocapid[0])
    if mocap_id < 0:
        raise ValueError(f"Body {MOCAP_NAME!r} is not a mocap body.")
    hand_body_id = int(model.body(HAND_BODY_NAME).id)

    # Component / slot / grasp-weld registries. Slots are discovered from the
    # model so adding a slot in the XML automatically extends the system.
    component_body_ids: dict[str, int] = {}
    component_weld_ids: dict[str, int] = {}
    for instance_name in COMPONENT_INSTANCES:
        try:
            component_body_ids[instance_name] = int(model.body(instance_name).id)
            component_weld_ids[instance_name] = int(model.equality(f"grasp_{instance_name}").id)
        except KeyError:
            pass
    slot_site_ids: dict[str, int] = {}
    for i in range(model.nsite):
        name = model.site(i).name
        if name and name.startswith("slot_"):
            slot_site_ids[name] = int(i)
    shared.update(
        available_components=sorted(component_body_ids.keys()),
        available_slots=sorted(slot_site_ids.keys()),
    )

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
        site_mat_zero_spin: np.ndarray,
        target_mat: np.ndarray,
        unwrap_reference_rad: float,
    ) -> float:
        # 5D arm IK intentionally ignores the final twist about the tool/object Z axis.
        # Given the zero-spin site frame (commanded arm posture with wrist_spin = 0),
        # this returns the absolute hinge angle that rotates the frame into the mocap
        # target's x/y phase around the target z axis.
        #
        # The principal angle is unwrapped near the commanded setpoint (not the measured
        # hinge angle) so that physical tracking lag can never flip the 2*pi branch.
        z_target = normalize_vec(np.asarray(target_mat, dtype=float).reshape(3, 3)[:, 2])
        desired_principal = signed_twist_about_axis(site_mat_zero_spin, target_mat, z_target)
        return unwrap_angle_near(desired_principal, unwrap_reference_rad)

    def move_toward(value: float, target: float, max_delta: float) -> float:
        return float(value + np.clip(target - value, -abs(max_delta), abs(max_delta)))

    if gripper_actuator_id is not None:
        data.ctrl[gripper_actuator_id] = GRIPPER_OPEN_CTRL
    mujoco.mj_forward(model, data)

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

    # ------------------------------------------------------------------
    # Flexible pick-place skill helpers.
    # ------------------------------------------------------------------
    def component_spec_of(instance: str) -> ComponentSpec:
        return COMPONENT_SPECS[COMPONENT_INSTANCES[instance]]

    def perceive_component(instance: str) -> tuple[np.ndarray, float]:
        """Simulated perception: read the instance pose from the physics state."""
        body_id = component_body_ids[instance]
        pos = np.array(data.body(body_id).xpos, copy=True)
        yaw = yaw_from_mat(np.asarray(data.body(body_id).xmat, dtype=float))
        return pos, yaw

    def slot_pose(slot: str) -> tuple[np.ndarray, float]:
        sid = slot_site_ids[slot]
        pos = np.array(data.site(sid).xpos, copy=True)
        yaw = yaw_from_mat(np.asarray(data.site(sid).xmat, dtype=float))
        return pos, yaw

    def attach_component(instance: str) -> dict[str, np.ndarray]:
        """Engage the pre-declared weld with the measured hand-object pose and
        record the site->object transform for the placement inverse map."""
        body_id = component_body_ids[instance]
        eq_id = component_weld_ids[instance]
        hand_pos = np.asarray(data.body(hand_body_id).xpos, dtype=float)
        hand_mat = np.asarray(data.body(hand_body_id).xmat, dtype=float).reshape(3, 3)
        obj_pos = np.asarray(data.body(body_id).xpos, dtype=float)
        obj_mat = np.asarray(data.body(body_id).xmat, dtype=float).reshape(3, 3)
        rel_pos = hand_mat.T @ (obj_pos - hand_pos)
        rel_quat = mat_to_quat(hand_mat.T @ obj_mat)
        model.eq_data[eq_id, :] = 0.0
        model.eq_data[eq_id, 3:6] = rel_pos
        model.eq_data[eq_id, 6:10] = rel_quat
        model.eq_data[eq_id, 10] = 1.0
        data.eq_active[eq_id] = 1
        site_pos = np.asarray(data.site(site_id).xpos, dtype=float)
        site_mat = np.asarray(data.site(site_id).xmat, dtype=float).reshape(3, 3)
        return {
            "pos_site_obj": site_mat.T @ (obj_pos - site_pos),
            "mat_site_obj": site_mat.T @ obj_mat,
        }

    def detach_component(instance: str) -> None:
        data.eq_active[component_weld_ids[instance]] = 0

    def build_pick_place_steps(task: PickPlaceTask, attach_record: dict[str, Any]) -> list[SkillStep]:
        """Parameterized six-leg trajectory. Pick legs are generated from the
        perceived component pose NOW; place legs are deferred factories that use
        the grasp transform captured at weld-engage time."""
        spec = component_spec_of(task.component)
        object_pos, object_yaw = perceive_component(task.component)
        grasp_pos, grasp_quat = compute_grasp_ee_pose(object_pos, object_yaw, spec)
        approach_pos = grasp_pos + np.asarray([0.0, 0.0, APPROACH_CLEARANCE])
        lift_pos = grasp_pos + np.asarray([0.0, 0.0, LIFT_HEIGHT])

        def place_ee_pose() -> tuple[np.ndarray, np.ndarray]:
            slot_pos, slot_yaw = slot_pose(task.slot)
            object_target_pos = slot_pos + np.asarray([0.0, 0.0, spec.half_height + spec.place_drop])
            return compute_place_ee_pose(
                object_target_pos,
                slot_yaw,
                attach_record["pos_site_obj"],
                attach_record["mat_site_obj"],
            )

        def place_pose_cmd() -> PoseCommand:
            pos, quat = place_ee_pose()
            return pose_command_from_position_quat(pos, quat)

        def seat_pose_cmd() -> PoseCommand:
            pos, quat = place_ee_pose()
            pos = pos - np.asarray([0.0, 0.0, float(spec.seat_press_m)])
            return pose_command_from_position_quat(pos, quat)

        def transfer_pose_cmd() -> PoseCommand:
            pos, quat = place_ee_pose()
            return pose_command_from_position_quat(pos + np.asarray([0.0, 0.0, APPROACH_CLEARANCE]), quat)

        steps = [
            SkillStep("approach", pose=pose_command_from_position_quat(approach_pos, grasp_quat)),
            SkillStep("descend", pose=pose_command_from_position_quat(grasp_pos, grasp_quat), pos_tol=0.003),
            SkillStep("grasp_close", pose=pose_command_from_position_quat(grasp_pos, grasp_quat), action="close_gripper", dwell_s=0.45),
            SkillStep("grasp_attach", pose=pose_command_from_position_quat(grasp_pos, grasp_quat), action="attach", dwell_s=0.15),
            SkillStep("lift", pose=pose_command_from_position_quat(lift_pos, grasp_quat)),
            SkillStep("transfer", pose_factory=transfer_pose_cmd),
            SkillStep("place", pose_factory=place_pose_cmd, pos_tol=0.003),
        ]
        if float(spec.seat_press_m) > 0.0:
            steps.append(SkillStep("seat_press", pose_factory=seat_pose_cmd, pos_tol=0.002, dwell_s=0.35))
        steps.extend(
            [
                SkillStep(
                    "release",
                    pose_factory=seat_pose_cmd if float(spec.seat_press_m) > 0.0 else place_pose_cmd,
                    action="release",
                    dwell_s=float(spec.release_dwell_s),
                ),
                SkillStep("retreat", pose_factory=transfer_pose_cmd, dwell_s=0.25),
            ]
        )
        return steps

    def resolve_order_entries(entries: list[dict[str, Any]], reserved: set[str]) -> list[PickPlaceTask]:
        """Turn order JSON entries into concrete tasks. Instances may be omitted:
        the first free instance of the requested type is auto-selected."""
        tasks: list[PickPlaceTask] = []
        for entry in entries:
            type_name = str(entry.get("type", ""))
            slot = str(entry.get("target_slot", ""))
            instance = str(entry.get("instance", "") or "")
            if slot not in slot_site_ids:
                raise ValueError(f"unknown slot: {slot}")
            if instance:
                if instance not in component_body_ids:
                    raise ValueError(f"unknown component instance: {instance}")
                if type_name and COMPONENT_INSTANCES.get(instance) != type_name:
                    raise ValueError(f"instance {instance} is not of type {type_name}")
            else:
                candidates = [
                    name for name, tname in COMPONENT_INSTANCES.items()
                    if tname == type_name and name in component_body_ids and name not in reserved
                ]
                if not candidates:
                    raise ValueError(f"no free instance of type {type_name}")
                instance = sorted(candidates)[0]
            reserved.add(instance)
            tasks.append(PickPlaceTask(component=instance, slot=slot))
        return tasks

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
        wrist_spin_target_rad = float(data.qpos[wrist_spin_qpos_id]) if wrist_spin_qpos_id is not None else 0.0
        # Persistent joint-space control command. While tracking, it re-anchors to the
        # measured qpos every frame (original convergent behavior). Once the 5D error
        # enters the hold deadband it is frozen bitwise, so wrist_spin reaction torque
        # cannot ratchet the seven arm joints away during pure-twist rotations.
        q_arm_ctrl_target = np.array(data.qpos[qpos_ids], copy=True)
        arm_in_deadband = False

        # Flexible pick-place skill state.
        gripper_ctrl_target = GRIPPER_OPEN_CTRL
        task_queue: list[PickPlaceTask] = []
        active_task: Optional[PickPlaceTask] = None
        active_steps: list[SkillStep] = []
        active_step_index = 0
        skill_phase = "move"  # "move" -> waiting for arrival, "dwell" -> action settling
        skill_dwell_until = 0.0
        skill_attach_record: dict[str, Any] = {}
        held_component = ""

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
                            # Manual jogging cancels any running pick-place task.
                            task_queue.clear()
                            active_task = None
                            active_steps = []
                            active_step_index = 0
                            skill_phase = "move"
                            motion.active_mode = str(command.get("mode", "move_one_step"))
                            motion.current_goal = waypoints[0]
                            motion.remaining_goals = waypoints[1:]
                            motion.active_goal_index = 1
                            motion.total_goal_count = len(waypoints)
                            set_mocap_pose(motion.current_goal)
                            status = f"start {motion.active_mode}: waypoint {motion.active_goal_index}/{motion.total_goal_count}"
                    elif ctype == "pick_place":
                        instance = str(command.get("component", ""))
                        slot = str(command.get("slot", ""))
                        if instance not in component_body_ids:
                            raise ValueError(f"unknown component: {instance}")
                        if slot not in slot_site_ids:
                            raise ValueError(f"unknown slot: {slot}")
                        motion.clear()
                        task_queue.append(PickPlaceTask(component=instance, slot=slot))
                        status = f"queued pick_place: {instance} -> {slot}"
                    elif ctype == "order":
                        entries = command.get("components", [])
                        reserved = {t.component for t in task_queue}
                        if active_task is not None:
                            reserved.add(active_task.component)
                        tasks = resolve_order_entries(list(entries), reserved)
                        motion.clear()
                        task_queue.extend(tasks)
                        status = f"order {command.get('order_id', '')} queued: {len(tasks)} tasks"
                    elif ctype == "stop":
                        motion.clear()
                        task_queue.clear()
                        active_task = None
                        active_steps = []
                        active_step_index = 0
                        skill_phase = "move"
                        joint_centering_active = joint_centering_enabled
                        arm_posture_reference = np.array(data.qpos[qpos_ids], copy=True)
                        q_arm_ctrl_target = np.array(data.qpos[qpos_ids], copy=True)
                        snap_mocap_to_site()
                        if wrist_spin_qpos_id is not None:
                            wrist_spin_target_rad = float(data.qpos[wrist_spin_qpos_id])
                        status = "stopped; tasks cleared; target snapped to current EE pose"
                    elif ctype == "ee_axes":
                        shared.show_ee_axes = bool(command.get("visible", True))
                    elif ctype == "mocap":
                        shared.show_mocap = bool(command.get("visible", True))
                    elif ctype == "offset":
                        shared.mocap_z_comp = float(command.get("mocap_z_comp", shared.mocap_z_comp))
                        snap_mocap_to_site()
                        q_arm_ctrl_target = np.array(data.qpos[qpos_ids], copy=True)
                        if wrist_spin_qpos_id is not None:
                            wrist_spin_target_rad = float(data.qpos[wrist_spin_qpos_id])
                        status = f"mocap z compensation = {shared.mocap_z_comp:.4f} m"
                    elif ctype == "quit":
                        running = False
                except Exception as exc:
                    status = f"command failed: {exc}"

            apply_visibility(ee_geom_ids, ee_site_ids, bool(shared.show_ee_axes))
            apply_visibility(mocap_geom_ids, mocap_site_ids, bool(shared.show_mocap))

            # ---- Flexible pick-place skill state machine ----
            if active_task is None and task_queue:
                active_task = task_queue.pop(0)
                skill_attach_record = {}
                try:
                    active_steps = build_pick_place_steps(active_task, skill_attach_record)
                    active_step_index = 0
                    skill_phase = "move"
                    set_mocap_pose(active_steps[0].resolve_pose())
                    status = f"task start: {active_task.component} -> {active_task.slot}"
                except Exception as exc:
                    status = f"task failed to start: {exc}"
                    active_task = None
                    active_steps = []

            if active_task is not None and active_steps:
                step = active_steps[active_step_index]
                if skill_phase == "move":
                    pos_err_step = float(np.linalg.norm(compensated_target_pos() - data.site(site_id).xpos))
                    ori_err_step = orientation_error()
                    if pos_err_step < float(step.pos_tol) and ori_err_step < float(step.ori_tol):
                        if step.action == "close_gripper":
                            gripper_ctrl_target = gripper_close_ctrl_for_width(component_spec_of(active_task.component).grip_width)
                        elif step.action == "attach":
                            skill_attach_record.update(attach_component(active_task.component))
                            held_component = active_task.component
                        elif step.action == "release":
                            detach_component(active_task.component)
                            gripper_ctrl_target = GRIPPER_OPEN_CTRL
                            held_component = ""
                        skill_phase = "dwell"
                        skill_dwell_until = time.time() + float(step.dwell_s)
                elif skill_phase == "dwell" and time.time() >= skill_dwell_until:
                    active_step_index += 1
                    skill_phase = "move"
                    if active_step_index >= len(active_steps):
                        status = f"task complete: {active_task.component} -> {active_task.slot}"
                        active_task = None
                        active_steps = []
                        active_step_index = 0
                        if not task_queue:
                            snap_mocap_to_site()
                    else:
                        set_mocap_pose(active_steps[active_step_index].resolve_pose())
                        status = f"{active_task.component} -> {active_task.slot}: {active_steps[active_step_index].label}"

            if gripper_actuator_id is not None:
                data.ctrl[gripper_actuator_id] = float(gripper_ctrl_target)

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
            #
            # Deadband with hysteresis: once the position and z-axis errors drop to
            # numerical noise level, freeze the seven arm joints completely; only wake
            # them up again when the error clearly leaves the noise floor (5x the entry
            # threshold, still 3-6x below the waypoint completion tolerances). A pure
            # twist of the mocap target around its own z axis then moves only wrist_spin
            # without dragging any FR3 joint. The wide hysteresis band also prevents a
            # micro limit cycle where tiny servo settling errors repeatedly wake the IK,
            # whose correction kick would re-excite the arm.
            q_arm = np.array(data.qpos[qpos_ids], copy=True)
            dx_norm = float(np.linalg.norm(dx))
            if arm_in_deadband:
                if dx_norm > 5.0 * float(args.arm_deadband_pos) or z_axis_error > 5.0 * float(args.arm_deadband_ori):
                    arm_in_deadband = False
            elif dx_norm < float(args.arm_deadband_pos) and z_axis_error < float(args.arm_deadband_ori):
                arm_in_deadband = True
            if arm_in_deadband:
                dq_arm = np.zeros(len(dof_ids), dtype=float)
            else:
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
                primary_scale = (dx_norm / max(float(args.position_tolerance), 1e-6)) + (z_axis_error / max(float(args.orientation_tolerance), 1e-6))
                pure_twist_or_steady = primary_scale < float(args.joint_secondary_gate)
                try:
                    jac5_dls = jac5.T @ np.linalg.solve(jac5 @ jac5.T + diag5, np.eye(5))
                    dq_arm = jac5_dls @ task5
                    nullspace_projector = eye_arm - jac5_dls @ jac5
                    if joint_centering_active:
                        if pure_twist_or_steady:
                            gamma_secondary = 0.0
                        else:
                            gamma_secondary = float(np.clip(primary_scale / max(float(args.joint_secondary_gate), 1e-6), 0.0, 1.0))
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
                # For the current XML the hinge axis, hand Z and attachment_site Z are
                # collinear and attachment_site's fixed local quat is a pure Z-phase, so
                # right-multiplying by Rz(-current_spin) reconstructs the zero-spin site
                # frame from the measured one.
                site_no_spin_mat = site_mat @ rot_z(-current_spin)
                desired_spin = compute_absolute_wrist_spin_target(site_no_spin_mat, target_mat, wrist_spin_target_rad)
                max_step = math.radians(float(args.wrist_spin_speed_deg_s)) * float(args.dt)
                wrist_spin_target_rad = move_toward(wrist_spin_target_rad, desired_spin, max_step)
                data.ctrl[wrist_spin_actuator_id] = float(wrist_spin_target_rad)

            dq_abs_max = float(np.max(np.abs(dq_arm))) if len(dq_arm) else 0.0
            if dq_abs_max > float(args.max_angvel):
                dq_arm *= float(args.max_angvel) / dq_abs_max

            # Arm command update:
            # - Outside the deadband the target is anchored to the measured joints each
            #   frame (classical resolved-rate stepping), preserving the original
            #   well-behaved convergence dynamics.
            # - Inside the deadband the persistent target is frozen bitwise. wrist_spin
            #   reaction torques may wiggle the measured joints, but the command never
            #   follows, so a pure mocap twist cannot ratchet the seven arm joints away.
            #   The anti-windup band only re-attaches the frozen target if the arm is
            #   ever pushed far away physically.
            if arm_in_deadband:
                q_arm_ctrl_target = np.clip(q_arm_ctrl_target, q_arm - SETPOINT_TRACKING_BAND, q_arm + SETPOINT_TRACKING_BAND)
            else:
                q_arm_ctrl_target = q_arm + dq_arm * float(args.dt)
            q_arm_ctrl_target = np.clip(q_arm_ctrl_target, joint_lower, joint_upper)
            data.ctrl[actuator_ids] = q_arm_ctrl_target
            mujoco.mj_step(model, data)

            viewer.sync()

            now = time.time()
            if now - last_state_time > 0.05:
                pose_rad = current_pose()
                pose_deg = np.concatenate([pose_rad[:3], np.rad2deg(pose_rad[3:])])
                pose_quat = current_pose_quat()[3:]
                q_arm_state = np.array(data.qpos[qpos_ids], copy=True)
                joint_margin, joint_center_cost = joint_centering_metrics(q_arm_state)
                shared.update(
                    pose_rad=[float(x) for x in pose_rad],
                    pose_deg=[float(x) for x in pose_deg],
                    pose_quat=[float(x) for x in pose_quat],
                    joint_angles_rad=[float(x) for x in q_arm_state],
                    joint_angles_deg=[float(x) for x in np.rad2deg(q_arm_state)],
                    joint_limit_margin=float(joint_margin),
                    joint_center_cost=float(joint_center_cost),
                    joint_centering_active=bool(joint_centering_active),
                    pos_err=float(np.linalg.norm(compensated_target_pos() - data.site(site_id).xpos)),
                    ori_err=orientation_error(),
                    status=status,
                    viewer_running=True,
                    waypoint_index=int(motion.active_goal_index),
                    waypoint_count=int(motion.total_goal_count),
                    active_mode=motion.active_mode or ("pick_place" if active_task is not None else "idle"),
                    skill_task=(f"{active_task.component} -> {active_task.slot}" if active_task is not None else ""),
                    skill_step_label=(
                        active_steps[active_step_index].label
                        if active_task is not None and active_step_index < len(active_steps)
                        else ""
                    ),
                    skill_step_index=int(active_step_index + 1) if active_task is not None else 0,
                    skill_step_count=int(len(active_steps)),
                    task_queue_len=int(len(task_queue)),
                    held_component=str(held_component),
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
    parser = argparse.ArgumentParser(description="Flexible FR3 pick-and-place cell: order-driven parameterized grasping onto an electrical mounting board")
    parser.add_argument("--ui-client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ui-python", type=str, default=None)
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--no-terminal-commands", action="store_true")
    parser.add_argument("--xml", type=str, default=str(Path(__file__).with_name("flexible_pick_place.xml")))
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--kpos", type=float, default=3.6)
    parser.add_argument("--kori", type=float, default=3.6)
    parser.add_argument("--kn", type=float, default=5.0)
    parser.add_argument("--joint-centering-gain", type=float, default=2.5, help="Global nullspace gain for joint-centering optimization used by all motion modes")
    parser.add_argument("--joint-limit-barrier-gain", type=float, default=0.04, help="Extra nullspace repulsion gain near joint limits")
    parser.add_argument("--max-nullspace-speed", type=float, default=4.0, help="Clamp for joint-limit barrier secondary velocity, rad/s")
    parser.add_argument("--joint-secondary-gate", type=float, default=0.25, help="Normalized 5D primary-error gate below which secondary optimization is frozen unless explicitly forced")
    parser.add_argument("--arm-deadband-pos", type=float, default=2e-4, help="Position deadband in meters; below it (together with the orientation deadband) the seven arm joints are frozen")
    parser.add_argument("--arm-deadband-ori", type=float, default=2e-3, help="Z-axis direction deadband in radians; below it (together with the position deadband) the seven arm joints are frozen")
    parser.add_argument("--wrist-spin-speed-deg-s", type=float, default=360.0, help="Maximum wrist_spin tracking speed in deg/s")
    parser.add_argument("--disable-joint-centering", action="store_true", help="Disable global nullspace joint-centering optimization")
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--max-angvel", type=float, default=21.0)
    parser.add_argument("--position-tolerance", type=float, default=0.003)
    parser.add_argument("--orientation-tolerance", type=float, default=0.03)
    parser.add_argument("--hide-ee-axes", action="store_true")
    parser.add_argument("--hide-mocap", action="store_true")
    parser.add_argument("--mocap-z-comp", type=float, default=0.0)
    parser.add_argument("--zero-gravity", action="store_true", help="Disable gravity (components will float; debugging only)")
    parser.add_argument("--gravity-comp", action="store_true", default=True, help="Enable robot body gravity compensation (default on)")
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
