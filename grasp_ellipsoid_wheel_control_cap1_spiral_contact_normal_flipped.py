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
  - lens-center world pose display and direct 6D pose reset
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
align_step1_fixed_pose_rad = np.asarray([0.620, 0.0, 0.310, -math.pi, 0.0, 0.5 * math.pi], dtype=float)
align_step1_fixed_pose_deg_text = "0.620000 0 0.310000 -180.00000 0 90.00000"
# Tangent center on the opposite wheel selected by locked-pose IK search:
# body center remains 0.200 m from the wheel axis and roll stays continuous
# for approximately 10..220 deg with pitch=-90 deg, yaw=0 deg.
optimized_polishing_contact_pose_deg = np.asarray(
    [0.000000, 0.529257, 0.439859, 90.0, -90.0, 0.0],
    dtype=float,
)
optimized_polishing_roll_range_deg = (10.0, 220.0)
locked_pose_ik_position_tolerance = 0.0010
locked_pose_ik_orientation_tolerance = 0.020
locked_pose_ik_orientation_weight = 0.20
locked_pose_min_joint_clearance = 0.080
locked_pose_ik_supplemental_seed_count = 24
locked_move_position_tolerance = 0.0010
locked_move_orientation_tolerance = 0.020
locked_move_intermediate_position_tolerance = 0.006
locked_move_intermediate_orientation_tolerance = 0.045
locked_move_transit_position_step = 0.015
locked_move_contact_position_step = 0.004
locked_move_body_angle_step = math.radians(5.0)
locked_move_in_place_translation_threshold = 0.003
locked_move_in_place_max_penetration = 0.0025
locked_move_wheel_retreat = 0.030
locked_move_reconfiguration_retreat = 0.060
locked_move_near_wheel_margin = 0.045
surface_scan_samples_per_side = 160
double_sphere_radius = 0.0725
double_sphere_center_half_distance = 0.0525
surface_scan_circle_radius = math.sqrt(
    max(0.0, double_sphere_radius * double_sphere_radius - double_sphere_center_half_distance * double_sphere_center_half_distance)
)
surface_scan_yaw_center = math.pi
surface_scan_step_time = 0.06
surface_scan_transition_lift = 0.040
# Cap1 Archimedean-spiral polishing scan defaults. Units are SI.
# cap1_sphere_center_offset is the distance from the lens mid-plane to cap1's
# mother-sphere center along cap1's body-fixed surface-axis normal. If your 52.5 mm is the total
# distance between the two mother-sphere centers, use 0.02625 here/in the UI.
# If it is already the half-distance used by the XML, keep 0.0525.
cap1_spiral_default_arc_step = 0.0020
cap1_spiral_default_radial_spacing = 0.0060
cap1_spiral_default_max_normal_angle_deg = 1.5
cap1_sphere_center_offset_default = double_sphere_center_half_distance
cap1_spiral_max_points = 2200

# Names expected in your FR3+gripper XML.
SITE_NAME = "attachment_site"
MOCAP_NAME = "target"
EE_AXIS_CYLINDER_GEOM_NAME = "ee_axis_z_center_cylinder"
ELLIPSOID_GEOM_NAME = "grasp_ellipsoid_geom"
WHEEL_CYLINDER_GEOM_NAME = "wheel_cylinder"
OPPOSITE_WHEEL_CYLINDER_GEOM_NAME = "wheel_cylinder_opposite"
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
ELLIPSOID_BODY_NAME = "grasp_ellipsoid"


@dataclass(frozen=True)
class PoseCommand:
    position: np.ndarray
    quat: np.ndarray
    raw_pose: np.ndarray  # x y z roll pitch yaw, radians
    posture_reference: Optional[np.ndarray] = None


@dataclass
class MotionState:
    current_goal: Optional[PoseCommand] = None
    remaining_goals: Optional[list[PoseCommand]] = None
    active_goal_index: int = 0
    total_goal_count: int = 0
    active_mode: str = ""
    protect_opposite_wheel_contact: bool = False

    def clear(self) -> None:
        self.current_goal = None
        self.remaining_goals = []
        self.active_goal_index = 0
        self.total_goal_count = 0
        self.active_mode = ""
        self.protect_opposite_wheel_contact = False


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
        self.lens_center_pose_rad: Optional[list[float]] = None
        self.lens_center_pose_deg: Optional[list[float]] = None
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
                "lens_center_pose_rad": None if self.lens_center_pose_rad is None else list(self.lens_center_pose_rad),
                "lens_center_pose_deg": None if self.lens_center_pose_deg is None else list(self.lens_center_pose_deg),
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
            elif self.path == "/lens_center_pose":
                with self.shared.lock:
                    lock_status = str(self.shared.workpiece_lock_status)
                if lock_status == "closing":
                    raise ValueError("待磨件正在锁定；请等待步骤2完成后再移动凸透镜中心。")
                pose = payload.get("pose", [])
                if not isinstance(pose, list) or len(pose) != 6:
                    raise ValueError("pose must have 6 elements: x y z roll pitch yaw")
                values = [float(x) for x in pose]
                self.shared.commands.put({"type": "lens_center_pose", "pose": values})
                if lock_status == "locked":
                    self.shared.update(status=f"请求按凸透镜中心目标反算机械臂关节运动: {format_pose(values)}")
                else:
                    self.shared.update(status=f"请求直接移动凸透镜中心: {format_pose(values)}")
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
            elif self.path == "/cap1_spiral_scan":
                with self.shared.lock:
                    lock_status = str(self.shared.workpiece_lock_status)
                if lock_status != "locked":
                    raise ValueError("待磨件尚未 locked；请先执行步骤2。")
                clearance = float(np.clip(float(payload.get("surface_clearance", 0.0)), -0.020, 0.020))
                arc_step = float(np.clip(float(payload.get("arc_step", cap1_spiral_default_arc_step)), 0.0005, 0.0100))
                radial_spacing = float(np.clip(float(payload.get("radial_spacing", cap1_spiral_default_radial_spacing)), 0.0010, 0.0300))
                max_normal_angle_deg = float(np.clip(float(payload.get("max_normal_angle_deg", cap1_spiral_default_max_normal_angle_deg)), 0.25, 10.0))
                center_offset = float(np.clip(float(payload.get("center_offset", cap1_sphere_center_offset_default)), 0.0, double_sphere_radius * 0.98))
                target_z_sign = 1.0 if float(payload.get("target_z_sign", 1.0)) >= 0.0 else -1.0
                self.shared.commands.put({
                    "type": "cap1_spiral_scan",
                    "surface_clearance": clearance,
                    "arc_step": arc_step,
                    "radial_spacing": radial_spacing,
                    "max_normal_angle_deg": max_normal_angle_deg,
                    "center_offset": center_offset,
                    "target_z_sign": target_z_sign,
                })
                self.shared.update(
                    status=(
                        f"请求执行 cap1 螺线打磨：arc_step={arc_step:.4f}m, "
                        f"radial_spacing={radial_spacing:.4f}m, normal≤{max_normal_angle_deg:.2f}°, "
                        f"target红轴/接触法线≈{' +X' if target_z_sign > 0 else ' -X'}，"
                        f"target蓝轴≈{' -Y' if target_z_sign > 0 else ' +Y'}"
                    ),
                    scan_status=f"waiting_start, clearance={clearance:.4f}m",
                )
                self._send_json({
                    "ok": True,
                    "surface_clearance": clearance,
                    "arc_step": arc_step,
                    "radial_spacing": radial_spacing,
                    "max_normal_angle_deg": max_normal_angle_deg,
                    "center_offset": center_offset,
                    "target_z_sign": target_z_sign,
                })
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
            self.last_lens_center_pose_rad: Optional[list[float]] = None
            self.last_lens_center_pose_deg: Optional[list[float]] = None
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

            title = QLabel("FR3 末端位姿 / 凸透镜中心位姿 / 函数轨迹 / Gripper 控制")
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
            copy_pose_btn = QPushButton("复制当前末端六维数组 (按 Move 单位)")
            copy_pose_btn.clicked.connect(self.copy_current_pose_array)
            pose_layout.addWidget(copy_pose_btn)
            layout.addWidget(pose_group)

            lens_center_group = QGroupBox("凸透镜中心六维坐标")
            lens_center_layout = QVBoxLayout(lens_center_group)
            lens_center_layout.addWidget(QLabel("rad:  x y z roll pitch yaw"))
            self.lens_center_pose_rad_label = QLabel("等待仿真数据...")
            self.lens_center_pose_rad_label.setObjectName("MonoLabel")
            lens_center_layout.addWidget(self.lens_center_pose_rad_label)
            lens_center_layout.addWidget(QLabel("deg:  x y z roll pitch yaw"))
            self.lens_center_pose_deg_label = QLabel("等待仿真数据...")
            self.lens_center_pose_deg_label.setObjectName("MonoLabel")
            lens_center_layout.addWidget(self.lens_center_pose_deg_label)

            lens_center_row = QHBoxLayout()
            lens_center_row.addWidget(QLabel("输入姿态角单位："))
            self.lens_center_unit_combo = QComboBox()
            self.lens_center_unit_combo.addItems(["rad", "deg"])
            self.lens_center_unit_combo.setCurrentText("deg")
            lens_center_row.addWidget(self.lens_center_unit_combo)
            lens_center_row.addStretch(1)
            lens_center_layout.addLayout(lens_center_row)

            self.lens_center_pose_boxes: dict[str, QDoubleSpinBox] = {}
            lens_center_grid = QGridLayout()
            lens_center_defaults = {
                name: float(value)
                for name, value in zip(
                    ("x", "y", "z", "roll", "pitch", "yaw"),
                    optimized_polishing_contact_pose_deg,
                )
            }
            for col, name in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
                box = QDoubleSpinBox()
                box.setDecimals(6)
                box.setRange(-10000, 10000)
                box.setSingleStep(0.01 if name in {"x", "y", "z"} else 0.05)
                box.setValue(lens_center_defaults[name])
                self.lens_center_pose_boxes[name] = box
                lens_center_grid.addWidget(QLabel(name), 0, col)
                lens_center_grid.addWidget(box, 1, col)
            lens_center_layout.addLayout(lens_center_grid)
            recommended_contact_label = QLabel(
                "推荐相切打磨中心: "
                f"[{optimized_polishing_contact_pose_deg[0]:.6f}, "
                f"{optimized_polishing_contact_pose_deg[1]:.6f}, "
                f"{optimized_polishing_contact_pose_deg[2]:.6f}] m; "
                f"roll 连续预检范围约 {optimized_polishing_roll_range_deg[0]:.0f}.."
                f"{optimized_polishing_roll_range_deg[1]:.0f} deg"
            )
            recommended_contact_label.setObjectName("MonoLabel")
            recommended_contact_label.setWordWrap(True)
            lens_center_layout.addWidget(recommended_contact_label)

            lens_center_array_row = QHBoxLayout()
            self.lens_center_array_input = QLineEdit()
            self.lens_center_array_input.setPlaceholderText("[x, y, z, roll, pitch, yaw]")
            lens_center_array_row.addWidget(QLabel("六维数组："))
            lens_center_array_row.addWidget(self.lens_center_array_input, 1)
            fill_lens_center_array_btn = QPushButton("从数组填入")
            fill_lens_center_array_btn.clicked.connect(self.fill_lens_center_pose_from_array)
            lens_center_array_row.addWidget(fill_lens_center_array_btn)
            lens_center_layout.addLayout(lens_center_array_row)

            lens_center_btns = QHBoxLayout()
            fill_lens_center_btn = QPushButton("将当前中心填入")
            fill_lens_center_btn.clicked.connect(self.fill_current_lens_center_pose)
            copy_lens_center_btn = QPushButton("复制当前中心六维数组")
            copy_lens_center_btn.clicked.connect(self.copy_current_lens_center_pose_array)
            fill_lens_center_move_btn = QPushButton("填入Move框")
            fill_lens_center_move_btn.clicked.connect(self.fill_lens_center_pose_to_move_box)
            move_lens_center_btn = QPushButton("移动凸透镜中心到该位姿")
            move_lens_center_btn.clicked.connect(self.move_lens_center_pose)
            lens_center_btns.addWidget(fill_lens_center_btn)
            lens_center_btns.addWidget(copy_lens_center_btn)
            lens_center_btns.addWidget(fill_lens_center_move_btn)
            lens_center_btns.addWidget(move_lens_center_btn)
            lens_center_btns.addStretch(1)
            lens_center_layout.addLayout(lens_center_btns)
            layout.addWidget(lens_center_group)

            align_group = QGroupBox("待磨件把手对齐")
            align_layout = QVBoxLayout(align_group)
            align_btns = QHBoxLayout()
            align_step1_btn = QPushButton("执行对齐步骤1")
            align_step1_btn.clicked.connect(self.align_workpiece_handle_step1)
            align_step2_btn = QPushButton("执行对齐步骤2")
            align_step2_btn.clicked.connect(self.align_workpiece_handle_step2)
            scan_btn = QPushButton("执行圆周扫描")
            scan_btn.clicked.connect(self.execute_ellipsoid_surface_scan)
            cap1_spiral_btn = QPushButton("执行cap1螺线打磨")
            cap1_spiral_btn.clicked.connect(self.execute_cap1_spiral_scan)
            align_btns.addWidget(align_step1_btn)
            align_btns.addWidget(align_step2_btn)
            align_btns.addWidget(scan_btn)
            align_btns.addWidget(cap1_spiral_btn)
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
            clearance_row.addWidget(QLabel("arc_step(m):"))
            self.cap1_arc_step_box = QDoubleSpinBox()
            self.cap1_arc_step_box.setDecimals(6)
            self.cap1_arc_step_box.setRange(0.0005, 0.0100)
            self.cap1_arc_step_box.setSingleStep(0.0005)
            self.cap1_arc_step_box.setValue(cap1_spiral_default_arc_step)
            clearance_row.addWidget(self.cap1_arc_step_box)
            clearance_row.addWidget(QLabel("loop_spacing(m):"))
            self.cap1_radial_spacing_box = QDoubleSpinBox()
            self.cap1_radial_spacing_box.setDecimals(6)
            self.cap1_radial_spacing_box.setRange(0.0010, 0.0300)
            self.cap1_radial_spacing_box.setSingleStep(0.0010)
            self.cap1_radial_spacing_box.setValue(cap1_spiral_default_radial_spacing)
            clearance_row.addWidget(self.cap1_radial_spacing_box)
            clearance_row.addStretch(1)
            align_layout.addLayout(clearance_row)

            cap1_row = QHBoxLayout()
            cap1_row.addWidget(QLabel("max_normal_angle(deg):"))
            self.cap1_normal_angle_box = QDoubleSpinBox()
            self.cap1_normal_angle_box.setDecimals(3)
            self.cap1_normal_angle_box.setRange(0.25, 10.0)
            self.cap1_normal_angle_box.setSingleStep(0.25)
            self.cap1_normal_angle_box.setValue(cap1_spiral_default_max_normal_angle_deg)
            cap1_row.addWidget(self.cap1_normal_angle_box)
            cap1_row.addWidget(QLabel("sphere_center_offset(m):"))
            self.cap1_center_offset_box = QDoubleSpinBox()
            self.cap1_center_offset_box.setDecimals(6)
            self.cap1_center_offset_box.setRange(0.0, double_sphere_radius * 0.98)
            self.cap1_center_offset_box.setSingleStep(0.0010)
            self.cap1_center_offset_box.setValue(cap1_sphere_center_offset_default)
            cap1_row.addWidget(self.cap1_center_offset_box)
            cap1_row.addWidget(QLabel("cap1姿态:"))
            self.cap1_target_z_combo = QComboBox()
            self.cap1_target_z_combo.addItems(["红轴接触+X / 蓝轴-Y", "红轴接触-X / 蓝轴+Y"])
            self.cap1_target_z_combo.setCurrentIndex(0)
            cap1_row.addWidget(self.cap1_target_z_combo)
            cap1_row.addStretch(1)
            align_layout.addLayout(cap1_row)
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

            one_array_row = QHBoxLayout()
            self.one_pose_array_input = QLineEdit()
            self.one_pose_array_input.setPlaceholderText("[x, y, z, roll, pitch, yaw]")
            one_array_row.addWidget(QLabel("六维数组："))
            one_array_row.addWidget(self.one_pose_array_input, 1)
            fill_one_array_btn = QPushButton("从数组填入")
            fill_one_array_btn.clicked.connect(self.fill_move_pose_from_array)
            one_array_row.addWidget(fill_one_array_btn)
            one_layout.addLayout(one_array_row)

            row = QHBoxLayout()
            btn = QPushButton("移动到该位姿")
            btn.clicked.connect(self.move_one_step)
            row.addWidget(btn)
            btn2 = QPushButton("把当前位姿填入")
            btn2.clicked.connect(self.fill_current_pose)
            row.addWidget(btn2)
            copy_move_pose_btn = QPushButton("复制当前六维数组")
            copy_move_pose_btn.clicked.connect(self.copy_current_pose_array)
            row.addWidget(copy_move_pose_btn)
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

        @staticmethod
        def pose_array_text(pose_rad: list[float], unit: str) -> str:
            vals = list(pose_rad)
            if unit == "deg":
                vals[3:] = [math.degrees(v) for v in vals[3:]]
            return "[" + ", ".join(f"{float(v):.6f}" for v in vals) + "]"

        @staticmethod
        def parse_pose_array(text: str) -> list[float]:
            raw = text.strip().replace("，", ",")
            if len(raw) >= 2 and ((raw[0] == "[" and raw[-1] == "]") or (raw[0] == "(" and raw[-1] == ")")):
                raw = raw[1:-1].strip()
            fields = [part for part in raw.replace(",", " ").split() if part]
            if len(fields) != 6:
                raise ValueError("请输入包含 6 个数值的数组，例如 [0.45, 0, 0.45, 3.141593, 0, 0]。")
            try:
                return [float(part) for part in fields]
            except ValueError as exc:
                raise ValueError("六维数组中包含无法解析的数值。") from exc

        @staticmethod
        def copy_text_to_system_clipboard(text: str) -> None:
            if sys.platform == "darwin" and shutil.which("pbcopy") is not None:
                subprocess.run(["/usr/bin/pbcopy"], input=text, text=True, check=True)
                return
            QApplication.clipboard().setText(text)
            QApplication.processEvents()

        def copy_current_pose_array(self) -> None:
            unit = self.unit()
            text = self.pose_array_text(self.last_pose_rad, unit)
            try:
                self.copy_text_to_system_clipboard(text)
            except (OSError, subprocess.CalledProcessError) as exc:
                QMessageBox.critical(self, "复制失败", f"无法写入系统剪贴板：{exc}")
                return
            self.status_label.setText(f"已复制当前末端六维数组（角度单位 {unit}）：{text}")

        def fill_move_pose_from_array(self) -> None:
            try:
                vals = self.parse_pose_array(self.one_pose_array_input.text())
            except ValueError as exc:
                QMessageBox.warning(self, "六维数组格式错误", str(exc))
                return
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), vals):
                self.pose_boxes[name].setValue(float(value))
            self.status_label.setText(f"已从六维数组填入 Move One Step（角度单位 {self.unit()}）。")

        def fill_current_pose(self) -> None:
            vals = list(self.last_pose_rad)
            if self.unit() == "deg":
                vals[3:] = [math.degrees(v) for v in vals[3:]]
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), vals):
                self.pose_boxes[name].setValue(float(value))
            self.status_label.setText("已把当前位姿填入 Move One Step。")

        def lens_center_unit(self) -> str:
            return "deg" if self.lens_center_unit_combo.currentText() == "deg" else "rad"

        def fill_current_lens_center_pose(self) -> None:
            if self.last_lens_center_pose_rad is None:
                self.status_label.setText("当前模型中未找到凸透镜中心，无法填入中心位姿输入框。")
                return
            vals = list(self.last_lens_center_pose_rad)
            if self.lens_center_unit() == "deg":
                vals[3:] = [math.degrees(v) for v in vals[3:]]
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), vals):
                self.lens_center_pose_boxes[name].setValue(float(value))
            self.status_label.setText("已把当前凸透镜中心位姿填入输入框。")

        def copy_current_lens_center_pose_array(self) -> None:
            if self.last_lens_center_pose_rad is None:
                self.status_label.setText("当前模型中未找到凸透镜中心，无法复制六维数组。")
                return
            unit = self.lens_center_unit()
            text = self.pose_array_text(self.last_lens_center_pose_rad, unit)
            try:
                self.copy_text_to_system_clipboard(text)
            except (OSError, subprocess.CalledProcessError) as exc:
                QMessageBox.critical(self, "复制失败", f"无法写入系统剪贴板：{exc}")
                return
            self.status_label.setText(f"已复制当前凸透镜中心六维数组（角度单位 {unit}）：{text}")

        def fill_lens_center_pose_from_array(self) -> None:
            try:
                vals = self.parse_pose_array(self.lens_center_array_input.text())
            except ValueError as exc:
                QMessageBox.warning(self, "六维数组格式错误", str(exc))
                return
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), vals):
                self.lens_center_pose_boxes[name].setValue(float(value))
            self.status_label.setText(f"已从六维数组填入凸透镜中心目标（角度单位 {self.lens_center_unit()}）。")

        def fill_lens_center_pose_to_move_box(self) -> None:
            if self.last_lens_center_pose_rad is None:
                self.status_label.setText("当前模型中未找到凸透镜中心，无法填入 Move One Step。")
                return
            vals = list(self.last_lens_center_pose_rad)
            if self.unit() == "deg":
                vals[3:] = [math.degrees(v) for v in vals[3:]]
            for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), vals):
                self.pose_boxes[name].setValue(float(value))
            self.status_label.setText("已把当前凸透镜中心位姿填入 Move One Step。")

        def get_lens_center_pose_rad(self) -> list[float]:
            vals = [self.lens_center_pose_boxes[n].value() for n in ("x", "y", "z", "roll", "pitch", "yaw")]
            if self.lens_center_unit() == "deg":
                vals[3:] = [math.radians(v) for v in vals[3:]]
            return [float(v) for v in vals]

        def move_lens_center_pose(self) -> None:
            self.post("/lens_center_pose", {"pose": self.get_lens_center_pose_rad()})

        def align_workpiece_handle_step1(self) -> None:
            self.post("/align_workpiece_handle_step1", {})

        def align_workpiece_handle_step2(self) -> None:
            self.post("/align_workpiece_handle_step2", {})

        def execute_ellipsoid_surface_scan(self) -> None:
            self.post("/ellipsoid_surface_scan", {"surface_clearance": float(self.surface_clearance_box.value())})

        def execute_cap1_spiral_scan(self) -> None:
            self.post(
                "/cap1_spiral_scan",
                {
                    "surface_clearance": float(self.surface_clearance_box.value()),
                    "arc_step": float(self.cap1_arc_step_box.value()),
                    "radial_spacing": float(self.cap1_radial_spacing_box.value()),
                    "max_normal_angle_deg": float(self.cap1_normal_angle_box.value()),
                    "center_offset": float(self.cap1_center_offset_box.value()),
                    "target_z_sign": 1.0 if self.cap1_target_z_combo.currentIndex() == 0 else -1.0,
                },
            )

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
                lens_center_pose_rad = state.get("lens_center_pose_rad")
                lens_center_pose_deg = state.get("lens_center_pose_deg")
                if isinstance(lens_center_pose_rad, list) and len(lens_center_pose_rad) == 6:
                    self.last_lens_center_pose_rad = [float(x) for x in lens_center_pose_rad]
                    self.lens_center_pose_rad_label.setText(format_pose(self.last_lens_center_pose_rad))
                else:
                    self.last_lens_center_pose_rad = None
                    self.lens_center_pose_rad_label.setText("当前模型中未找到凸透镜中心。")
                if isinstance(lens_center_pose_deg, list) and len(lens_center_pose_deg) == 6:
                    self.last_lens_center_pose_deg = [float(x) for x in lens_center_pose_deg]
                    self.lens_center_pose_deg_label.setText(format_pose(self.last_lens_center_pose_deg))
                else:
                    self.last_lens_center_pose_deg = None
                    self.lens_center_pose_deg_label.setText("当前模型中未找到凸透镜中心。")
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
    wheel_spin_opposite_actuator_id: Optional[int] = None
    try:
        wheel_spin_opposite_actuator_id = model.actuator("wheel_spin_opposite").id
    except KeyError:
        wheel_spin_opposite_actuator_id = None
    wheel_spin_opposite_mirror_actuator_id: Optional[int] = None
    try:
        wheel_spin_opposite_mirror_actuator_id = model.actuator("wheel_spin_opposite_mirror").id
    except KeyError:
        wheel_spin_opposite_mirror_actuator_id = None
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

    ellipsoid_control: Optional[tuple[int, int, int]] = None
    try:
        body_id = int(model.body(ELLIPSOID_BODY_NAME).id)
    except KeyError:
        body_id = -1
    if body_id >= 0:
        # The workpiece is virtual and must remain suspended until explicitly moved or locked.
        model.body_gravcomp[body_id] = 1.0
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
    ellipsoid_geom_id = maybe_geom(ELLIPSOID_GEOM_NAME)
    wheel_cylinder_geom_id = maybe_geom(WHEEL_CYLINDER_GEOM_NAME)
    opposite_wheel_cylinder_geom_id = maybe_geom(OPPOSITE_WHEEL_CYLINDER_GEOM_NAME)
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

    def current_lens_center_pose() -> tuple[Optional[list[float]], Optional[list[float]]]:
        if ellipsoid_control is None:
            return None, None
        body_id, _, _ = ellipsoid_control
        pose_rad = np.concatenate(
            [
                np.array(data.xpos[body_id], copy=True),
                rpy_from_mat(data.xmat[body_id]),
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

    def pose_command_with_posture_reference(cmd: PoseCommand, posture_reference: np.ndarray) -> PoseCommand:
        return PoseCommand(
            position=np.array(cmd.position, copy=True),
            quat=np.array(cmd.quat, copy=True),
            raw_pose=np.array(cmd.raw_pose, copy=True),
            posture_reference=np.array(posture_reference, copy=True),
        )

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

    def body_command_from_pose(position: np.ndarray, mat: np.ndarray) -> PoseCommand:
        return pose_command_from_position_mat(np.asarray(position, dtype=float), np.asarray(mat, dtype=float))

    def current_lens_body_command() -> PoseCommand:
        if ellipsoid_control is None:
            raise ValueError("当前模型中没有可移动的待磨件。")
        body_id, _, _ = ellipsoid_control
        return body_command_from_pose(np.array(data.xpos[body_id], copy=True), np.asarray(data.xmat[body_id]).reshape(3, 3))

    def command_rotation_distance(start: PoseCommand, end: PoseCommand) -> float:
        dot = abs(float(np.dot(normalize_vector(start.quat, "start quaternion"), normalize_vector(end.quat, "end quaternion"))))
        return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))

    def interpolate_body_commands(start: PoseCommand, end: PoseCommand, position_step: float) -> list[PoseCommand]:
        distance = float(np.linalg.norm(end.position - start.position))
        angle = command_rotation_distance(start, end)
        count = max(
            1,
            int(math.ceil(distance / position_step)),
            int(math.ceil(angle / locked_move_body_angle_step)),
        )
        return interpolate_commands(start, end, count)

    def build_locked_lens_center_move_goals(
        values: list[float] | np.ndarray,
        lock_state: WorkpieceLockState,
    ) -> tuple[list[PoseCommand], bool, bool]:
        target_body_pos, target_body_mat = body_pose_from_lens_center_pose(values)
        start_body = current_lens_body_command()
        target_body = body_command_from_pose(target_body_pos, target_body_mat)
        requested_translation = float(np.linalg.norm(target_body.position - start_body.position))
        in_place_orientation_change = requested_translation <= locked_move_in_place_translation_threshold
        if in_place_orientation_change:
            # Preserve the user's entered center exactly while changing
            # orientation; only the residual tracking error is corrected.
            start_body = body_command_from_pose(target_body.position, mat_from_quat(start_body.quat))
        stages: list[tuple[PoseCommand, float]] = [(target_body, locked_move_transit_position_step)]
        uses_wheel_detour = False

        if opposite_wheel_cylinder_geom_id is not None and not in_place_orientation_change:
            wheel_center = np.array(data.geom_xpos[opposite_wheel_cylinder_geom_id], copy=True)
            wheel_radius = float(model.geom_size[opposite_wheel_cylinder_geom_id][0])
            wheel_axis = normalize_vector(
                np.asarray(data.geom_xmat[opposite_wheel_cylinder_geom_id], dtype=float).reshape(3, 3)[:, 2],
                "opposite wheel axis",
            )
            radial = target_body.position - wheel_center
            radial -= wheel_axis * float(np.dot(radial, wheel_axis))
            radial_distance = float(np.linalg.norm(radial))
            nominal_contact_distance = wheel_radius + surface_scan_circle_radius
            if radial_distance > 1e-8 and radial_distance <= nominal_contact_distance + locked_move_near_wheel_margin:
                outward = radial / radial_distance
                retreat = outward * locked_move_wheel_retreat
                retreat_start = body_command_from_pose(start_body.position + retreat, mat_from_quat(start_body.quat))
                transit_target = body_command_from_pose(target_body.position + retreat, mat_from_quat(start_body.quat))
                retreat_target = body_command_from_pose(target_body.position + retreat, target_body_mat)
                # Keep the gripped workpiece orientation unchanged during a long
                # transfer. For an in-place polishing rotation, stages remains a
                # fixed-center orientation interpolation so contact is preserved.
                stages = [
                    (retreat_start, locked_move_transit_position_step),
                    (transit_target, locked_move_transit_position_step),
                    (retreat_target, locked_move_transit_position_step),
                    (target_body, locked_move_contact_position_step),
                ]
                uses_wheel_detour = True

        body_commands: list[PoseCommand] = []
        previous = start_body
        for stage, position_step in stages:
            body_commands.extend(interpolate_body_commands(previous, stage, position_step))
            previous = stage
        return (
            [site_command_from_body_pose(cmd.position, mat_from_quat(cmd.quat), lock_state) for cmd in body_commands],
            uses_wheel_detour,
            in_place_orientation_change,
        )

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

    def tangent_hint_for_normal(normal: np.ndarray, preferred: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        n = normalize_vector(normal, "surface normal")
        t = np.asarray(preferred, dtype=float) - n * float(np.dot(n, preferred))
        if float(np.linalg.norm(t)) < 1e-8:
            t = np.asarray(fallback, dtype=float) - n * float(np.dot(n, fallback))
        return normalize_vector(t, "surface tangent")

    def frame_from_normal_tangent(normal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        n = normalize_vector(normal, "surface normal")
        t = np.asarray(tangent, dtype=float) - n * float(np.dot(n, tangent))
        t = normalize_vector(t, "surface tangent")
        b = normalize_vector(np.cross(n, t), "surface binormal")
        # Columns are tangent, binormal, normal.
        return np.column_stack([t, b, n])

    def body_mat_from_cap_normal(
        local_normal: np.ndarray,
        world_normal: np.ndarray,
        local_tangent_hint: np.ndarray,
        world_tangent_hint: np.ndarray,
    ) -> np.ndarray:
        local_frame = frame_from_normal_tangent(local_normal, local_tangent_hint)
        world_frame = frame_from_normal_tangent(world_normal, world_tangent_hint)
        return world_frame @ local_frame.T

    def cap1_frame_to_body_mat(lock_state: WorkpieceLockState) -> np.ndarray:
        """Return cap1 coordinates expressed in the lens body frame.

        The polishing surface is part of the lens, not of the gripper. Rotating
        the EE around the handle axis during step 1 changes site_to_body_mat but
        must not rotate cap1 within the lens. With the current body-frame
        definition, cap1 uses:
          cap x = -body Y axis
          cap y = -body Z axis
          cap z = +body X axis, the surface-axis normal
        """
        if lock_state.mode != "locked" or lock_state.site_to_body_mat is None:
            raise ValueError("待磨件尚未 locked，无法建立 cap1 面坐标系。")
        return np.column_stack(
            [
                np.asarray([0.0, -1.0, 0.0], dtype=float),
                np.asarray([0.0, 0.0, -1.0], dtype=float),
                np.asarray([1.0, 0.0, 0.0], dtype=float),
            ]
        )

    def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = normalize_vector(axis, "rotation axis")
        x, y, z = axis
        c = math.cos(angle)
        s = math.sin(angle)
        C = 1.0 - c
        return np.asarray(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=float,
        )

    def spin_body_mat_to_align_site_z(
        body_mat: np.ndarray,
        contact_normal_world: np.ndarray,
        preferred_site_z_world: np.ndarray,
        lock_state: WorkpieceLockState,
    ) -> np.ndarray:
        """Use cap-normal spin freedom to keep the target frame reachable.

        `body_mat` already maps the sampled cap normal to `contact_normal_world`.
        Premultiplying by a rotation around `contact_normal_world` preserves that
        normal alignment while changing the locked EE/site frame orientation.
        """
        if lock_state.site_to_body_mat is None:
            return body_mat
        axis = normalize_vector(contact_normal_world, "contact normal")
        desired = normalize_vector(preferred_site_z_world, "preferred target blue axis")
        site_mat = np.asarray(body_mat, dtype=float) @ lock_state.site_to_body_mat.T
        current_site_z = normalize_vector(site_mat[:, 2], "current target blue axis")

        current_proj = current_site_z - axis * float(np.dot(current_site_z, axis))
        desired_proj = desired - axis * float(np.dot(desired, axis))
        if float(np.linalg.norm(current_proj)) < 1e-8 or float(np.linalg.norm(desired_proj)) < 1e-8:
            return body_mat
        current_proj = normalize_vector(current_proj, "projected target blue axis")
        desired_proj = normalize_vector(desired_proj, "projected preferred target blue axis")
        angle = math.atan2(float(np.dot(axis, np.cross(current_proj, desired_proj))), float(np.dot(current_proj, desired_proj)))
        return rotation_matrix_from_axis_angle(axis, angle) @ np.asarray(body_mat, dtype=float)

    def build_cap1_spiral_scan_goals(
        clearance: float,
        arc_step: float,
        radial_spacing: float,
        max_normal_angle_deg: float,
        center_offset: float,
        target_z_sign: float,
        lock_state: WorkpieceLockState,
    ) -> tuple[list[PoseCommand], list[str]]:
        """Build an Archimedean-spiral polishing path on cap1.

        cap1 is modeled in a polishing frame fixed to the lens body: cap x/y
        span the widest circular plane and cap z is its surface-axis normal.
        This remains valid if step 1 changes the EE's grip yaw. The path samples
        an Archimedean spiral in the
        cap xOy projection, then projects each point to the spherical cap. The
        step size is measured on the spherical surface, capped by the requested
        adjacent-normal angle. Each sampled cap normal is mapped to the
        wheel-side contact normal and converted into an EE target through the
        locked workpiece transform. The cap normal constraint leaves one free
        spin around the contact normal; choose that spin so the final red
        target/mocap blue z-axis stays on the reachable side instead of flipping.
        """
        if wheel_cylinder_geom_id is None:
            raise ValueError(f"当前模型中没有 {WHEEL_CYLINDER_GEOM_NAME}。")
        if lock_state.mode != "locked":
            raise ValueError("待磨件尚未 locked；请先执行步骤2。")

        radius = float(double_sphere_radius)
        center_offset = float(np.clip(center_offset, 0.0, radius * 0.98))
        rho_max = math.sqrt(max(0.0, radius * radius - center_offset * center_offset))
        if rho_max < 1e-6:
            raise ValueError("cap1 可扫描半径过小；请检查 sphere_center_offset。")

        # Bound surface arc step by the desired adjacent-normal change. On a
        # sphere, normal-angle change is exactly surface arc length / radius.
        normal_limited_step = radius * math.radians(max(0.1, float(max_normal_angle_deg)))
        target_ds = float(np.clip(min(float(arc_step), normal_limited_step), 0.0002, 0.0200))
        loop_spacing = float(np.clip(radial_spacing, 0.0010, 0.0300))
        spiral_b = loop_spacing / (2.0 * math.pi)

        cap_to_body = cap1_frame_to_body_mat(lock_state)
        local_normals: list[np.ndarray] = []
        cap_normals: list[np.ndarray] = []
        theta = 0.0
        while True:
            rho = min(rho_max, spiral_b * theta)
            x = rho * math.cos(theta)
            y = rho * math.sin(theta)
            z_rel = math.sqrt(max(0.0, radius * radius - rho * rho))
            cap_normal = normalize_vector(np.asarray([x, y, z_rel], dtype=float), "cap1 cap-frame normal")
            cap_normals.append(cap_normal)
            local_normals.append(cap_to_body @ cap_normal)
            if rho >= rho_max - 1e-9:
                break
            ds_dtheta = math.sqrt((radius * radius / max(1e-12, radius * radius - rho * rho)) * spiral_b * spiral_b + rho * rho)
            dtheta = target_ds / max(ds_dtheta, 1e-9)
            # Keep the final edge region from being under-sampled.
            dtheta = float(np.clip(dtheta, 1e-4, 0.35))
            theta += dtheta
            if len(local_normals) >= cap1_spiral_max_points:
                break

        if len(local_normals) < 2:
            raise ValueError("cap1 螺线采样点不足。")

        wheel_center = np.array(data.geom_xpos[wheel_cylinder_geom_id], copy=True)
        wheel_radius = float(model.geom_size[wheel_cylinder_geom_id][0])
        side_dir = np.asarray([1.0, 0.0, 0.0], dtype=float)

        # cap1's surface normal is fixed in the lens body. The remaining spin is
        # used only to keep the locked EE target blue axis on a stable +/-Y side
        # after the body-fixed cap normal is aligned to the wheel.
        contact_normal_world = normalize_vector(
            side_dir * (1.0 if float(target_z_sign) >= 0.0 else -1.0),
            "contact normal",
        )
        preferred_site_z_world = normalize_vector(
            np.asarray([0.0, -1.0 if float(target_z_sign) >= 0.0 else 1.0, 0.0], dtype=float),
            "preferred target blue axis",
        )

        contact_point_world = wheel_center + side_dir * (wheel_radius + float(clearance))
        cap_sphere_center_world = contact_point_world - radius * contact_normal_world
        cap_sphere_center_local = cap_to_body @ np.asarray([0.0, 0.0, -center_offset], dtype=float)
        world_tangent_hint = tangent_hint_for_normal(contact_normal_world, np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 1.0, 0.0]))

        commands: list[PoseCommand] = []
        labels: list[str] = []
        max_observed_angle_deg = 0.0
        min_site_z_dot = 1.0
        min_site_x_dot = 1.0
        min_normal_dot = 1.0
        for i, local_normal in enumerate(local_normals):
            cap_tangent_hint = tangent_hint_for_normal(
                cap_normals[i],
                np.asarray([0.0, 1.0, 0.0], dtype=float),
                np.asarray([1.0, 0.0, 0.0], dtype=float),
            )
            local_tangent_hint = cap_to_body @ cap_tangent_hint
            base_body_mat = body_mat_from_cap_normal(
                local_normal=local_normal,
                world_normal=contact_normal_world,
                local_tangent_hint=local_tangent_hint,
                world_tangent_hint=world_tangent_hint,
            )
            body_mat = spin_body_mat_to_align_site_z(
                base_body_mat,
                contact_normal_world,
                preferred_site_z_world,
                lock_state,
            )

            body_pos = cap_sphere_center_world - body_mat @ cap_sphere_center_local
            cmd = site_command_from_body_pose(body_pos, body_mat, lock_state)
            site_mat = mat_from_quat(cmd.quat)
            site_x_dot = float(np.dot(site_mat[:, 0], contact_normal_world))
            site_z_dot = float(np.dot(site_mat[:, 2], preferred_site_z_world))
            normal_dot = float(np.dot(body_mat @ local_normal, contact_normal_world))
            commands.append(cmd)
            min_site_x_dot = min(min_site_x_dot, site_x_dot)
            min_site_z_dot = min(min_site_z_dot, site_z_dot)
            min_normal_dot = min(min_normal_dot, normal_dot)
            if i > 0:
                dot = float(np.clip(np.dot(local_normals[i - 1], local_normal), -1.0, 1.0))
                max_observed_angle_deg = max(max_observed_angle_deg, math.degrees(math.acos(dot)))
            labels.append(
                f"cap1 spiral, normal≤{max_observed_angle_deg:.2f}deg, "
                f"cap_n·contact≥{min_normal_dot:.2f}, red_x·contact≥{min_site_x_dot:.2f}, "
                f"blue_z·pref≥{min_site_z_dot:.2f}"
            )

        # A small lift after the scan reduces the chance of staying pressed into
        # the wheel when the final point is reached.
        final_lift = pose_command_from_position_quat(commands[-1].position + np.asarray([0.0, 0.0, surface_scan_transition_lift]), commands[-1].quat)
        commands.append(final_lift)
        labels.append("cap1 spiral finished lift")
        return commands, labels

    def body_pose_from_lens_center_pose(values: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cmd = pose_array_to_command(values)
        return cmd.position, mat_from_quat(cmd.quat)

    def site_command_from_lens_center_pose(values: list[float] | np.ndarray, lock_state: WorkpieceLockState) -> PoseCommand:
        body_pos, body_mat = body_pose_from_lens_center_pose(values)
        return site_command_from_body_pose(body_pos, body_mat, lock_state)

    def set_lens_center_pose(values: list[float] | np.ndarray) -> None:
        body_pos, body_mat = body_pose_from_lens_center_pose(values)
        apply_workpiece_body_pose(body_pos, body_mat)

    def capture_workpiece_lock(lock_state: WorkpieceLockState) -> None:
        if ellipsoid_control is None:
            raise ValueError("当前模型中没有可移动的 ellipsoid，无法锁定待磨件。")
        body_id, _, _ = ellipsoid_control
        site_pos = np.array(data.site(site_id).xpos, copy=True)
        site_mat = np.asarray(data.site(site_id).xmat, dtype=float).reshape(3, 3)
        body_pos = np.array(data.xpos[body_id], copy=True)
        body_mat = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        # Capture the closed-gripper relationship from the actual poses. This
        # automatically respects the step-1 handle-axis yaw (currently +90 deg).
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

    def locked_opposite_wheel_clearance(lock_state: WorkpieceLockState) -> Optional[float]:
        """Return radial surface clearance before writing the virtual workpiece pose."""
        if (
            opposite_wheel_cylinder_geom_id is None
            or lock_state.mode != "locked"
            or lock_state.site_to_body_pos is None
        ):
            return None
        site_pos = np.array(data.site(site_id).xpos, copy=True)
        site_mat = np.asarray(data.site(site_id).xmat, dtype=float).reshape(3, 3)
        body_pos = site_pos + site_mat @ lock_state.site_to_body_pos
        wheel_center = np.array(data.geom_xpos[opposite_wheel_cylinder_geom_id], copy=True)
        wheel_axis = normalize_vector(
            np.asarray(data.geom_xmat[opposite_wheel_cylinder_geom_id], dtype=float).reshape(3, 3)[:, 2],
            "opposite wheel axis",
        )
        radial = body_pos - wheel_center
        radial -= wheel_axis * float(np.dot(radial, wheel_axis))
        wheel_radius = float(model.geom_size[opposite_wheel_cylinder_geom_id][0])
        return float(np.linalg.norm(radial) - (wheel_radius + surface_scan_circle_radius))

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

    def solve_locked_pose_posture(goal: PoseCommand, starting_q: Optional[np.ndarray] = None) -> np.ndarray:
        """Find a joint-limit-safe IK branch before commanding a locked workpiece pose."""
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise ValueError("locked 凸透镜中心移动需要 scipy 进行可达性预检查。") from exc

        target_pos = goal.position + compensation_vec()
        target_quat = np.asarray(goal.quat, dtype=float)
        q_lower = np.asarray([model.jnt_range[jid][0] for jid in joint_ids], dtype=float)
        q_upper = np.asarray([model.jnt_range[jid][1] for jid in joint_ids], dtype=float)
        current_q = np.array(
            data.qpos[qpos_ids] if starting_q is None else starting_q,
            copy=True,
        )
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        ctrl_backup = np.array(data.ctrl, copy=True)
        candidates: list[tuple[float, float, float, float, np.ndarray]] = []

        def residual(q_arm: np.ndarray) -> np.ndarray:
            data.qpos[qpos_ids] = q_arm
            mujoco.mj_forward(model, data)
            trial_quat = np.zeros(4, dtype=float)
            trial_quat_conj = np.zeros(4, dtype=float)
            error = np.zeros(4, dtype=float)
            omega = np.zeros(3, dtype=float)
            mujoco.mju_mat2Quat(trial_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(trial_quat_conj, trial_quat)
            mujoco.mju_mulQuat(error, target_quat, trial_quat_conj)
            mujoco.mju_quat2Vel(omega, error, 1.0)
            return np.concatenate(
                [
                    target_pos - data.site(site_id).xpos,
                    locked_pose_ik_orientation_weight * omega,
                ]
            )

        try:
            # Euler targets around pitch=-90 deg admit multiple IK branches.
            # A solve from only the current pose/home pose can converge to a
            # joint-limit endpoint that looks reachable but cannot satisfy the
            # strict final target tolerance. Use deterministic supplemental
            # seeds so identical UI commands remain reproducible.
            rng = np.random.default_rng(20260525)
            supplemental_seeds = [
                rng.uniform(q_lower + 1e-3, q_upper - 1e-3)
                for _ in range(locked_pose_ik_supplemental_seed_count)
            ]
            seeds = [current_q, q0_arm, *supplemental_seeds]
            for seed in seeds:
                result = least_squares(
                    residual,
                    np.clip(seed, q_lower + 1e-7, q_upper - 1e-7),
                    bounds=(q_lower, q_upper),
                    max_nfev=500,
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                )
                error = residual(result.x)
                pos_error = float(np.linalg.norm(error[:3]))
                ori_error = float(np.linalg.norm(error[3:]) / locked_pose_ik_orientation_weight)
                if (
                    pos_error <= locked_pose_ik_position_tolerance
                    and ori_error <= locked_pose_ik_orientation_tolerance
                ):
                    joint_clearance = float(np.min(np.minimum(result.x - q_lower, q_upper - result.x)))
                    travel = float(np.linalg.norm(result.x - current_q))
                    candidates.append((joint_clearance, travel, pos_error, ori_error, np.array(result.x, copy=True)))
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            data.ctrl[:] = ctrl_backup
            mujoco.mj_forward(model, data)

        if not candidates:
            raise ValueError(
                "目标会使末端 IK 撞到关节限位或无法满足误差；未移动。"
                "请调整凸透镜中心位置或避开当前极限姿态。"
            )
        comfortably_clear = [item for item in candidates if item[0] >= locked_pose_min_joint_clearance]
        if comfortably_clear:
            comfortably_clear.sort(key=lambda item: (item[1], -item[0], item[2], item[3]))
            return comfortably_clear[0][4]
        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        return candidates[0][4]

    def solve_continuous_path_postures(
        goals: list[PoseCommand],
        initial_q: Optional[np.ndarray] = None,
        failure_message: str = "接触原位旋转",
    ) -> list[PoseCommand]:
        """Solve adjacent targets from the previous solution without changing IK branch."""
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise ValueError("locked 凸透镜中心移动需要 scipy 进行可达性预检查。") from exc

        q_lower = np.asarray([model.jnt_range[jid][0] for jid in joint_ids], dtype=float)
        q_upper = np.asarray([model.jnt_range[jid][1] for jid in joint_ids], dtype=float)
        q_seed = np.array(
            data.qpos[qpos_ids] if initial_q is None else initial_q,
            copy=True,
        )
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        ctrl_backup = np.array(data.ctrl, copy=True)
        solved_goals: list[PoseCommand] = []
        try:
            for index, goal in enumerate(goals, start=1):
                target_pos = goal.position + compensation_vec()
                target_quat = np.asarray(goal.quat, dtype=float)

                def residual(q_arm: np.ndarray) -> np.ndarray:
                    data.qpos[qpos_ids] = q_arm
                    mujoco.mj_forward(model, data)
                    trial_quat = np.zeros(4, dtype=float)
                    trial_quat_conj = np.zeros(4, dtype=float)
                    error = np.zeros(4, dtype=float)
                    omega = np.zeros(3, dtype=float)
                    mujoco.mju_mat2Quat(trial_quat, data.site(site_id).xmat)
                    mujoco.mju_negQuat(trial_quat_conj, trial_quat)
                    mujoco.mju_mulQuat(error, target_quat, trial_quat_conj)
                    mujoco.mju_quat2Vel(omega, error, 1.0)
                    return np.concatenate(
                        [
                            target_pos - data.site(site_id).xpos,
                            locked_pose_ik_orientation_weight * omega,
                        ]
                    )

                result = least_squares(
                    residual,
                    np.clip(q_seed, q_lower + 1e-7, q_upper - 1e-7),
                    bounds=(q_lower, q_upper),
                    max_nfev=500,
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                )
                error = residual(result.x)
                pos_error = float(np.linalg.norm(error[:3]))
                ori_error = float(np.linalg.norm(error[3:]) / locked_pose_ik_orientation_weight)
                if (
                    pos_error > locked_pose_ik_position_tolerance
                    or ori_error > locked_pose_ik_orientation_tolerance
                ):
                    raise ValueError(
                        f"{failure_message}在 waypoint {index}/{len(goals)} 已超出当前连续 IK 分支；"
                        "为避免待磨件穿入打磨轮，命令未执行。请先远离打磨轮后再调整姿态。"
                    )
                q_seed = np.array(result.x, copy=True)
                solved_goals.append(pose_command_with_posture_reference(goal, q_seed))
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            data.ctrl[:] = ctrl_backup
            mujoco.mj_forward(model, data)
        return solved_goals

    def validate_in_place_continuous_path(goals: list[PoseCommand]) -> None:
        solve_continuous_path_postures(goals)

    def build_contact_reconfiguration_goals(
        values: list[float] | np.ndarray,
        lock_state: WorkpieceLockState,
    ) -> list[PoseCommand]:
        """Move away from the wheel, change IK branch, then return to contact."""
        if opposite_wheel_cylinder_geom_id is None:
            raise ValueError("当前模型中没有用于安全换支的第二个打磨轮。")
        target_body_pos, target_body_mat = body_pose_from_lens_center_pose(values)
        start_body = current_lens_body_command()
        target_body = body_command_from_pose(target_body_pos, target_body_mat)
        wheel_center = np.array(data.geom_xpos[opposite_wheel_cylinder_geom_id], copy=True)
        wheel_axis = normalize_vector(
            np.asarray(data.geom_xmat[opposite_wheel_cylinder_geom_id], dtype=float).reshape(3, 3)[:, 2],
            "opposite wheel axis",
        )
        radial = target_body.position - wheel_center
        radial -= wheel_axis * float(np.dot(radial, wheel_axis))
        outward = normalize_vector(radial, "workpiece-to-opposite-wheel radial")
        away_position = target_body.position + outward * locked_move_reconfiguration_retreat
        away_start = body_command_from_pose(away_position, mat_from_quat(start_body.quat))
        away_target = body_command_from_pose(away_position, target_body_mat)

        retreat_body_goals = interpolate_body_commands(
            start_body,
            away_start,
            locked_move_transit_position_step,
        )
        retreat_goals = [
            site_command_from_body_pose(cmd.position, mat_from_quat(cmd.quat), lock_state)
            for cmd in retreat_body_goals
        ]
        retreat_goals = solve_continuous_path_postures(
            retreat_goals,
            failure_message="安全撤离",
        )
        retreat_posture = np.array(retreat_goals[-1].posture_reference, copy=True)

        turn_body_goals = interpolate_body_commands(
            away_start,
            away_target,
            locked_move_transit_position_step,
        )
        turn_goals = [
            site_command_from_body_pose(cmd.position, mat_from_quat(cmd.quat), lock_state)
            for cmd in turn_body_goals
        ]
        turn_posture = solve_locked_pose_posture(turn_goals[-1], starting_q=retreat_posture)
        turn_goals = [
            pose_command_with_posture_reference(cmd, turn_posture)
            for cmd in turn_goals
        ]

        approach_body_goals = interpolate_body_commands(
            away_target,
            target_body,
            locked_move_contact_position_step,
        )
        approach_goals = [
            site_command_from_body_pose(cmd.position, mat_from_quat(cmd.quat), lock_state)
            for cmd in approach_body_goals
        ]
        approach_posture = solve_locked_pose_posture(approach_goals[-1], starting_q=turn_posture)
        approach_goals = [
            pose_command_with_posture_reference(cmd, approach_posture)
            for cmd in approach_goals
        ]
        return [*retreat_goals, *turn_goals, *approach_goals]

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
        arm_posture_reference = q0_arm.copy()
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
        status = "就绪：可在 UI 中执行函数轨迹。"
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
            nonlocal stream_waypoints, stream_index, stream_last_time, stream_started, stream_wait_until, surface_scan_waypoints, arm_posture_reference
            stream_waypoints = []
            stream_index = 0
            stream_last_time = 0.0
            stream_started = False
            stream_wait_until = 0.0
            surface_scan.clear()
            surface_scan_waypoints = []
            motion.clear()
            arm_posture_reference = q0_arm.copy()

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
                            motion.total_goal_count = len(waypoints)

                            if mode == "function_path":
                                # Function path is streamed in time instead of waiting for each
                                # sampled waypoint to be exactly reached. Keep the visible mocap
                                # at its current pose during the hold period, then send point 1.
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
                                    f"随后每 {stream_step_time:.3f}s 推进一个点"
                                )
                            else:
                                # move_one_step still waits until the target is reached.
                                stream_waypoints = []
                                stream_index = 0
                                motion.current_goal = waypoints[0]
                                motion.active_goal_index = 1
                                motion.remaining_goals = waypoints[1:]
                                set_mocap_pose(motion.current_goal)
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
                    elif ctype == "cap1_spiral_scan":
                        clearance = float(np.clip(float(command.get("surface_clearance", 0.0)), -0.020, 0.020))
                        arc_step = float(np.clip(float(command.get("arc_step", cap1_spiral_default_arc_step)), 0.0005, 0.0100))
                        radial_spacing = float(np.clip(float(command.get("radial_spacing", cap1_spiral_default_radial_spacing)), 0.0010, 0.0300))
                        max_normal_angle_deg = float(np.clip(float(command.get("max_normal_angle_deg", cap1_spiral_default_max_normal_angle_deg)), 0.25, 10.0))
                        center_offset = float(np.clip(float(command.get("center_offset", cap1_sphere_center_offset_default)), 0.0, double_sphere_radius * 0.98))
                        target_z_sign = 1.0 if float(command.get("target_z_sign", 1.0)) >= 0.0 else -1.0
                        if workpiece_lock.mode != "locked":
                            scan_status_text = "inactive"
                            status = "cap1 螺线打磨失败：待磨件尚未 locked；请先执行步骤2。"
                        else:
                            goals, labels = build_cap1_spiral_scan_goals(
                                clearance=clearance,
                                arc_step=arc_step,
                                radial_spacing=radial_spacing,
                                max_normal_angle_deg=max_normal_angle_deg,
                                center_offset=center_offset,
                                target_z_sign=target_z_sign,
                                lock_state=workpiece_lock,
                            )
                            reset_motion_execution()
                            surface_scan_waypoints = goals
                            surface_scan.active = True
                            surface_scan.streaming = False
                            surface_scan.index = 0
                            surface_scan.clearance = clearance
                            surface_scan.labels = labels
                            motion.active_mode = "cap1_spiral_scan"
                            motion.current_goal = goals[0]
                            motion.remaining_goals = []
                            motion.active_goal_index = 1
                            motion.total_goal_count = len(goals)
                            set_mocap_pose(motion.current_goal)
                            scan_status_text = (
                                f"cap1_spiral waiting_start, point 1/{len(goals)}, "
                                f"clearance={clearance:.4f}m, arc_step={arc_step:.4f}m, "
                                f"target红轴/接触法线≈{' +X' if target_z_sign > 0 else ' -X'}, "
                                f"target蓝轴≈{' -Y' if target_z_sign > 0 else ' +Y'}"
                            )
                            status = f"cap1 螺线打磨准备中：共 {len(goals)} 点，等待第一个点到达；target 红轴法线和蓝轴方向已按选项生成。"
                    elif ctype == "align_workpiece_handle_step2":
                        if ellipsoid_control is None:
                            workpiece_lock.clear()
                            status = "对齐步骤2失败：当前模型中没有可移动的待磨件 freejoint。"
                        else:
                            # Step 2 should cancel the step-1 fixed-pose motion. Otherwise
                            # active_mode can remain "align_step1_fixed", blocking the
                            # locked workpiece mocap drag logic after the lock is captured.
                            reset_motion_execution()
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
                        surface_scan.clear()
                        surface_scan_waypoints = []
                        scan_status_text = "inactive"
                        arm_posture_reference = q0_arm.copy()
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
                    elif ctype == "lens_center_pose":
                        pose_values = command.get("pose", [])
                        if workpiece_lock.mode == "closing":
                            status = "待磨件正在锁定；请等待步骤2完成后再移动凸透镜中心。"
                        elif workpiece_lock.mode == "locked":
                            goals, uses_wheel_detour, in_place_orientation_change = build_locked_lens_center_move_goals(
                                pose_values,
                                workpiece_lock,
                            )
                            if not goals:
                                raise ValueError("凸透镜中心移动未生成有效目标。")
                            automatic_reconfiguration = False
                            if in_place_orientation_change:
                                try:
                                    validate_in_place_continuous_path(goals)
                                except ValueError:
                                    goals = build_contact_reconfiguration_goals(pose_values, workpiece_lock)
                                    automatic_reconfiguration = True
                                    uses_wheel_detour = True
                            if automatic_reconfiguration and goals[0].posture_reference is not None:
                                safe_posture = np.array(goals[0].posture_reference, copy=True)
                            else:
                                safe_posture = solve_locked_pose_posture(goals[-1])
                            reset_motion_execution()
                            arm_posture_reference = safe_posture
                            scan_status_text = "inactive"
                            motion.active_mode = "locked_lens_center_pose"
                            motion.protect_opposite_wheel_contact = in_place_orientation_change
                            motion.current_goal = goals[0]
                            motion.remaining_goals = goals[1:]
                            motion.active_goal_index = 1
                            motion.total_goal_count = len(goals)
                            set_mocap_pose(motion.current_goal)
                            status = (
                                (
                                    "locked：原位旋转超出当前连续 IK 范围，正在自动撤离 60mm、换支并回靠；"
                                    if automatic_reconfiguration
                                    else
                                    "locked：接触中原位旋转已通过连续分支预检查，并启用防穿入监控；"
                                    if in_place_orientation_change
                                    else "locked：靠近打磨轮，正在安全撤离/转向/回靠；"
                                    if uses_wheel_detour
                                    else "locked："
                                )
                                + "已选择关节限位安全的 IK 分支，并按凸透镜中心目标移动到 "
                                f"{format_pose(pose_values)}"
                            )
                        else:
                            set_lens_center_pose(pose_values)
                            status = f"已直接移动凸透镜中心: {format_pose(pose_values)}"
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
            if wheel_spin_actuator_id is not None:
                lo, hi = model.actuator_ctrlrange[wheel_spin_actuator_id]
                data.ctrl[wheel_spin_actuator_id] = float(np.clip(2.0, lo, hi))
            if wheel_spin_opposite_actuator_id is not None:
                lo, hi = model.actuator_ctrlrange[wheel_spin_opposite_actuator_id]
                data.ctrl[wheel_spin_opposite_actuator_id] = float(np.clip(2.0, lo, hi))
            if wheel_spin_opposite_mirror_actuator_id is not None:
                lo, hi = model.actuator_ctrlrange[wheel_spin_opposite_mirror_actuator_id]
                data.ctrl[wheel_spin_opposite_mirror_actuator_id] = float(np.clip(2.0, lo, hi))

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

                if motion.active_mode in {"ellipsoid_surface_scan", "cap1_spiral_scan"} and surface_scan.active and surface_scan_waypoints:
                    now_for_scan = time.time()
                    if not surface_scan.streaming:
                        if pos_err_goal < args.position_tolerance and ori_err_goal < args.orientation_tolerance:
                            surface_scan.streaming = True
                            surface_scan.last_step_at = now_for_scan
                            scan_status_text = (
                                f"{surface_scan.label()}, point {surface_scan.index + 1}/{len(surface_scan_waypoints)}, "
                                f"clearance={surface_scan.clearance:.4f}m"
                            )
                            status = f"表面扫描开始：{scan_status_text}"
                        else:
                            scan_status_text = (
                                f"waiting_start, point 1/{len(surface_scan_waypoints)}, "
                                f"clearance={surface_scan.clearance:.4f}m"
                            )
                            status = f"表面扫描准备中：等待第一个点到达。{scan_status_text}"
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
                            status = f"表面扫描执行中：{scan_status_text}"
                    else:
                        if pos_err_goal < args.position_tolerance and ori_err_goal < args.orientation_tolerance:
                            scan_status_text = (
                                f"finished, point {surface_scan.index + 1}/{len(surface_scan_waypoints)}, "
                                f"clearance={surface_scan.clearance:.4f}m"
                            )
                            status = f"表面扫描完成：{scan_status_text}"
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
                    if motion.active_mode == "locked_lens_center_pose":
                        final_locked_goal = not motion.remaining_goals
                        position_tolerance = (
                            locked_move_position_tolerance
                            if final_locked_goal
                            else locked_move_intermediate_position_tolerance
                        )
                        orientation_tolerance = (
                            locked_move_orientation_tolerance
                            if final_locked_goal
                            else locked_move_intermediate_orientation_tolerance
                        )
                    else:
                        position_tolerance = args.position_tolerance
                        orientation_tolerance = args.orientation_tolerance
                    reached_goal = pos_err_goal < position_tolerance and ori_err_goal < orientation_tolerance
                    if reached_goal:
                        if motion.remaining_goals:
                            motion.current_goal = motion.remaining_goals.pop(0)
                            motion.active_goal_index += 1
                            if motion.current_goal.posture_reference is not None:
                                arm_posture_reference = np.array(motion.current_goal.posture_reference, copy=True)
                            set_mocap_pose(motion.current_goal)
                            status = f"继续执行 waypoint {motion.active_goal_index}/{motion.total_goal_count}"
                        else:
                            if motion.active_mode == "align_step1_fixed":
                                status = f"对齐步骤1执行完成：已移动到固定目标 {align_step1_fixed_pose_deg_text}"
                            elif motion.active_mode == "locked_lens_center_pose":
                                status = "locked 凸透镜中心目标执行完成：机械臂已移动到反算末端目标。"
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
                dq_arm += nullspace_projector @ (Kn * (arm_posture_reference - q_arm))
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
            guarded_arm_qpos = (
                np.array(data.qpos[qpos_ids], copy=True)
                if motion.protect_opposite_wheel_contact
                else None
            )
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

            if motion.protect_opposite_wheel_contact and guarded_arm_qpos is not None:
                clearance = locked_opposite_wheel_clearance(workpiece_lock)
                if clearance is not None and clearance < -locked_move_in_place_max_penetration:
                    data.qpos[qpos_ids] = guarded_arm_qpos
                    data.qvel[dof_ids] = 0.0
                    data.ctrl[actuator_ids] = guarded_arm_qpos
                    mujoco.mj_forward(model, data)
                    snap_mocap_to_site()
                    motion.clear()
                    arm_posture_reference = np.array(guarded_arm_qpos, copy=True)
                    status = (
                        "接触原位旋转已安全停止：预测待磨件进入打磨轮 "
                        f"{-clearance * 1000.0:.2f}mm，超过允许值 "
                        f"{locked_move_in_place_max_penetration * 1000.0:.2f}mm。"
                    )
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
                lens_center_pose_rad, lens_center_pose_deg = current_lens_center_pose()
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
                    lens_center_pose_rad=lens_center_pose_rad,
                    lens_center_pose_deg=lens_center_pose_deg,
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
