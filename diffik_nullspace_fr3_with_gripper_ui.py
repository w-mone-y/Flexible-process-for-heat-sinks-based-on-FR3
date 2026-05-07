#!/usr/bin/env python3
"""
FR3 differential IK controller with a PySide6 UI on macOS.

Why this version exists:
- On macOS, MuJoCo viewer is usually launched with `mjpython`.
- PySide6's QApplication can hang when created inside the same `mjpython` GUI process.
- This file therefore runs MuJoCo/viewer in the main mjpython process, and launches the
  PySide6 UI in a normal Python subprocess. The two processes communicate over localhost HTTP.

Run from your project root:
    mjpython diffik_nullspace_fr3_pyside6_ui_visibility.py --xml franka_fr3/scene_fr3_with_gripper_full_close_visual_toggle.xml

If the UI subprocess cannot import PySide6, install it in your active environment:
    pip install PySide6

If `python` does not point to your ai_learning environment, pass it explicitly:
    mjpython diffik_nullspace_fr3_pyside6_ui_visibility.py --ui-python /path/to/python --xml ...
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
from urllib import request as urllib_request

import mujoco
import mujoco.viewer
import numpy as np


# -------------------------------
# Controller parameters
# -------------------------------
integration_dt: float = 0.05
damping: float = 1e-5
Kpos: float = 1.0
Kori: float = 1.0
gravity_compensation: bool = True
dt: float = 0.002
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
max_angvel = 1.5

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
ARM_ACTUATOR_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
SITE_NAME = "attachment_site"
MOCAP_NAME = "target"
KEY_NAME = "home"
GRIPPER_ACTUATOR_CANDIDATES = ["gripper", "actuator8"]
FINGER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]


# -------------------------------
# Math helpers
# -------------------------------
def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX roll-pitch-yaw to MuJoCo quaternion [w, x, y, z]."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ],
        dtype=float,
    )


def mat_to_rpy(mat9: np.ndarray) -> np.ndarray:
    """Rotation matrix to ZYX roll-pitch-yaw."""
    R = np.asarray(mat9, dtype=float).reshape(3, 3)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def pose_from_site(data: mujoco.MjData, site_id: int) -> np.ndarray:
    pos = data.site(site_id).xpos.copy()
    rpy = mat_to_rpy(data.site(site_id).xmat)
    return np.concatenate([pos, rpy])


def pose_error_for_reached(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, target_pose: np.ndarray) -> tuple[float, float]:
    pos_err = float(np.linalg.norm(target_pose[:3] - data.site(site_id).xpos))
    target_quat = rpy_to_quat(*target_pose[3:])
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    err_quat = np.zeros(4)
    err_vel = np.zeros(3)
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(err_quat, target_quat, site_quat_conj)
    mujoco.mju_quat2Vel(err_vel, err_quat, 1.0)
    ori_err = float(np.linalg.norm(err_vel))
    return pos_err, ori_err


def parse_pose_from_json(payload: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(payload.get("pose", []), dtype=float)
    if pose.shape != (6,):
        raise ValueError("pose must contain exactly 6 numbers: x y z roll pitch yaw")
    unit = str(payload.get("unit", "rad")).lower()
    if unit == "deg":
        pose[3:] = np.deg2rad(pose[3:])
    elif unit != "rad":
        raise ValueError("unit must be 'rad' or 'deg'")
    return pose


def parse_poses_from_json(payload: dict[str, Any]) -> list[np.ndarray]:
    poses_raw = payload.get("poses", [])
    if not isinstance(poses_raw, list) or not poses_raw:
        raise ValueError("poses must be a non-empty list")
    unit = str(payload.get("unit", "rad")).lower()
    out: list[np.ndarray] = []
    for row in poses_raw:
        pose = np.asarray(row, dtype=float)
        if pose.shape != (6,):
            raise ValueError("each waypoint must contain exactly 6 numbers")
        if unit == "deg":
            pose[3:] = np.deg2rad(pose[3:])
        elif unit != "rad":
            raise ValueError("unit must be 'rad' or 'deg'")
        out.append(pose)
    return out


# -------------------------------
# Shared command/state
# -------------------------------
@dataclass
class SharedControlState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    current_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    target_queue: list[np.ndarray] = field(default_factory=list)
    active_target: Optional[np.ndarray] = None
    active: bool = False
    stop_requested: bool = False
    status: str = "空闲"
    gripper_value: float = 0.0
    show_ee_axes: bool = True
    show_mocap: bool = True
    pos_err: float = 0.0
    ori_err: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            pose = self.current_pose.copy()
            active_target = None if self.active_target is None else self.active_target.copy()
            return {
                "pose_rad": pose.tolist(),
                "pose_deg": np.concatenate([pose[:3], np.rad2deg(pose[3:])]).tolist(),
                "active_target_rad": None if active_target is None else active_target.tolist(),
                "active": self.active,
                "queue_len": len(self.target_queue),
                "status": self.status,
                "gripper": float(self.gripper_value),
                "show_ee_axes": bool(self.show_ee_axes),
                "show_mocap": bool(self.show_mocap),
                "pos_err": float(self.pos_err),
                "ori_err": float(self.ori_err),
            }


# -------------------------------
# HTTP server for UI <-> controller
# -------------------------------
def make_handler(shared: SharedControlState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FR3ControllerHTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep terminal clean.
            return

        def _send_json(self, obj: dict[str, Any], code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/state":
                self._send_json(shared.snapshot())
            else:
                self._send_json({"ok": False, "error": "unknown endpoint"}, 404)

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/move_one":
                    pose = parse_pose_from_json(payload)
                    with shared.lock:
                        shared.target_queue = []
                        shared.active_target = pose
                        shared.active = True
                        shared.stop_requested = False
                        shared.status = "执行 move_one_step"
                    self._send_json({"ok": True})
                elif self.path == "/move_multi":
                    poses = parse_poses_from_json(payload)
                    with shared.lock:
                        shared.active_target = poses[0]
                        shared.target_queue = poses[1:]
                        shared.active = True
                        shared.stop_requested = False
                        shared.status = f"执行 move_multi_step：1/{len(poses)}"
                    self._send_json({"ok": True, "count": len(poses)})
                elif self.path == "/stop":
                    with shared.lock:
                        shared.stop_requested = True
                        shared.target_queue = []
                        shared.active_target = None
                        shared.active = False
                        shared.status = "已请求停止"
                    self._send_json({"ok": True})
                elif self.path == "/gripper":
                    value = float(payload.get("value", 0.0))
                    value = max(0.0, min(255.0, value))
                    with shared.lock:
                        shared.gripper_value = value
                        shared.status = f"gripper = {value:.0f}"
                    self._send_json({"ok": True, "value": value})
                elif self.path == "/ee_axes":
                    visible = bool(payload.get("visible", True))
                    with shared.lock:
                        shared.show_ee_axes = visible
                        shared.status = "显示机械臂末端坐标轴" if visible else "隐藏机械臂末端坐标轴"
                    self._send_json({"ok": True, "visible": visible})
                elif self.path == "/mocap":
                    visible = bool(payload.get("visible", True))
                    with shared.lock:
                        shared.show_mocap = visible
                        shared.status = "显示 mocap/目标坐标轴" if visible else "隐藏 mocap/目标坐标轴"
                    self._send_json({"ok": True, "visible": visible})
                else:
                    self._send_json({"ok": False, "error": "unknown endpoint"}, 404)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, 400)

    return Handler


def start_http_server(shared: SharedControlState, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(shared))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def http_post(port: int, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=2.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get(port: int, path: str) -> dict[str, Any]:
    with urllib_request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


# -------------------------------
# PySide6 UI subprocess
# -------------------------------
def run_ui_client(port: int) -> int:
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QSlider,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # noqa: BLE001
        print("[UI] 无法导入 PySide6。请在 ai_learning 环境中运行：pip install PySide6", file=sys.stderr)
        print(f"[UI] 具体错误：{exc}", file=sys.stderr)
        return 2

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("FR3 PySide6 控制面板")
            self.setMinimumWidth(640)
            self._updating_gripper = False
            self.last_pose_rad = [0.0] * 6

            root = QWidget()
            layout = QVBoxLayout(root)
            self.setCentralWidget(root)

            # Current pose.
            pose_group = QGroupBox("当前末端位姿")
            pose_layout = QGridLayout(pose_group)
            self.pose_rad_label = QLabel("rad: --")
            self.pose_deg_label = QLabel("deg: --")
            self.err_label = QLabel("误差: --")
            self.status_label = QLabel("状态: 连接中...")
            pose_layout.addWidget(self.pose_rad_label, 0, 0)
            pose_layout.addWidget(self.pose_deg_label, 1, 0)
            pose_layout.addWidget(self.err_label, 2, 0)
            pose_layout.addWidget(self.status_label, 3, 0)
            layout.addWidget(pose_group)

            # One-step controls.
            one_group = QGroupBox("Move One Step")
            one_layout = QGridLayout(one_group)
            self.unit_combo = QComboBox()
            self.unit_combo.addItems(["rad", "deg"])
            one_layout.addWidget(QLabel("角度单位"), 0, 0)
            one_layout.addWidget(self.unit_combo, 0, 1)
            self.inputs: list[QLineEdit] = []
            names = ["x", "y", "z", "roll", "pitch", "yaw"]
            for i, name in enumerate(names):
                edit = QLineEdit()
                edit.setPlaceholderText(name)
                edit.setText("0.000")
                self.inputs.append(edit)
                one_layout.addWidget(QLabel(name), 1 + i // 3, (i % 3) * 2)
                one_layout.addWidget(edit, 1 + i // 3, (i % 3) * 2 + 1)
            fill_btn = QPushButton("填入当前位姿")
            move_btn = QPushButton("执行 Move One Step")
            stop_btn = QPushButton("停止当前任务")
            fill_btn.clicked.connect(self.fill_current)
            move_btn.clicked.connect(self.move_one)
            stop_btn.clicked.connect(self.stop)
            one_layout.addWidget(fill_btn, 3, 0, 1, 2)
            one_layout.addWidget(move_btn, 3, 2, 1, 2)
            one_layout.addWidget(stop_btn, 3, 4, 1, 2)
            layout.addWidget(one_group)

            # Multi-step controls.
            multi_group = QGroupBox("Move Multi Step：每行一个 waypoint：x y z roll pitch yaw")
            multi_layout = QVBoxLayout(multi_group)
            self.multi_text = QPlainTextEdit()
            self.multi_text.setPlaceholderText("0.45 0.00 0.45 3.14 0 0\n0.45 0.10 0.45 3.14 0 0")
            multi_btn = QPushButton("执行 Move Multi Step")
            multi_btn.clicked.connect(self.move_multi)
            multi_layout.addWidget(self.multi_text)
            multi_layout.addWidget(multi_btn)
            layout.addWidget(multi_group)

            # Gripper controls.
            grip_group = QGroupBox("Gripper")
            grip_layout = QHBoxLayout(grip_group)
            self.gripper_slider = QSlider(Qt.Orientation.Horizontal)
            self.gripper_slider.setRange(0, 255)
            self.gripper_spin = QSpinBox()
            self.gripper_spin.setRange(0, 255)
            self.gripper_slider.valueChanged.connect(self.gripper_spin.setValue)
            self.gripper_spin.valueChanged.connect(self.gripper_slider.setValue)
            self.gripper_slider.sliderReleased.connect(self.set_gripper_from_slider)
            self.gripper_spin.editingFinished.connect(self.set_gripper_from_spin)
            grip_layout.addWidget(QLabel("闭合 0"))
            grip_layout.addWidget(self.gripper_slider)
            grip_layout.addWidget(QLabel("张开 255"))
            grip_layout.addWidget(self.gripper_spin)
            layout.addWidget(grip_group)

            # Visualization toggles. These are mirrored to the terminal commands as well.
            vis_group = QGroupBox("显示 / 隐藏")
            vis_layout = QHBoxLayout(vis_group)
            self.ee_axes_checkbox = QCheckBox("显示机械臂末端坐标轴")
            self.ee_axes_checkbox.setChecked(True)
            self.mocap_checkbox = QCheckBox("显示 mocap 和 mocap 坐标轴")
            self.mocap_checkbox.setChecked(True)
            self.ee_axes_checkbox.toggled.connect(self.set_ee_axes_visibility)
            self.mocap_checkbox.toggled.connect(self.set_mocap_visibility)
            vis_layout.addWidget(self.ee_axes_checkbox)
            vis_layout.addWidget(self.mocap_checkbox)
            layout.addWidget(vis_group)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.poll_state)
            self.timer.start(100)

        def _pose_from_inputs(self) -> list[float]:
            try:
                return [float(edit.text().strip()) for edit in self.inputs]
            except ValueError as exc:
                raise ValueError("六维目标必须都是数字") from exc

        def _request_post(self, path: str, payload: dict[str, Any]) -> None:
            try:
                result = http_post(port, path, payload)
                if not result.get("ok", False):
                    raise RuntimeError(result.get("error", "未知错误"))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "请求失败", str(exc))

        def poll_state(self) -> None:
            try:
                state = http_get(port, "/state")
                pose_rad = state["pose_rad"]
                pose_deg = state["pose_deg"]
                self.last_pose_rad = pose_rad
                self.pose_rad_label.setText(
                    "rad: " + " ".join(f"{v:.4f}" for v in pose_rad)
                )
                self.pose_deg_label.setText(
                    "deg: " + " ".join(f"{v:.4f}" for v in pose_deg)
                )
                self.err_label.setText(
                    f"误差: pos={state.get('pos_err', 0.0):.5f} m, ori={state.get('ori_err', 0.0):.5f} rad"
                )
                self.status_label.setText(f"状态: {state.get('status', '')}")
                g = int(round(float(state.get("gripper", 0))))
                if not self.gripper_slider.isSliderDown() and self.gripper_spin.value() != g:
                    self.gripper_spin.setValue(g)

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
            except Exception as exc:  # noqa: BLE001
                self.status_label.setText(f"状态: 暂时无法连接控制器：{exc}")

        def fill_current(self) -> None:
            vals = self.last_pose_rad.copy()
            if self.unit_combo.currentText() == "deg":
                vals[3:] = list(np.rad2deg(vals[3:]))
            for edit, val in zip(self.inputs, vals):
                edit.setText(f"{val:.6f}")

        def move_one(self) -> None:
            try:
                pose = self._pose_from_inputs()
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "输入错误", str(exc))
                return
            self._request_post("/move_one", {"pose": pose, "unit": self.unit_combo.currentText()})

        def move_multi(self) -> None:
            try:
                poses: list[list[float]] = []
                for lineno, line in enumerate(self.multi_text.toPlainText().splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) != 6:
                        raise ValueError(f"第 {lineno} 行不是 6 个数字")
                    poses.append([float(x) for x in parts])
                if not poses:
                    raise ValueError("请至少输入一个 waypoint")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "输入错误", str(exc))
                return
            self._request_post("/move_multi", {"poses": poses, "unit": self.unit_combo.currentText()})

        def stop(self) -> None:
            self._request_post("/stop", {})

        def set_gripper_from_slider(self) -> None:
            self._request_post("/gripper", {"value": self.gripper_slider.value()})

        def set_gripper_from_spin(self) -> None:
            self._request_post("/gripper", {"value": self.gripper_spin.value()})

        def set_ee_axes_visibility(self, checked: bool) -> None:
            self._request_post("/ee_axes", {"visible": bool(checked)})

        def set_mocap_visibility(self, checked: bool) -> None:
            self._request_post("/mocap", {"visible": bool(checked)})

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


# -------------------------------
# MuJoCo controller process
# -------------------------------
def find_gripper_actuator_id(model: mujoco.MjModel) -> Optional[int]:
    for name in GRIPPER_ACTUATOR_CANDIDATES:
        try:
            return model.actuator(name).id
        except KeyError:
            continue
    return None


def set_finger_qpos_from_ctrl(model: mujoco.MjModel, data: mujoco.MjData, ctrl_value: float) -> None:
    finger_open = max(0.0, min(255.0, ctrl_value)) / 255.0 * 0.04
    for joint_name in FINGER_JOINT_NAMES:
        try:
            jid = model.joint(joint_name).id
            qpos_adr = int(model.jnt_qposadr[jid])
            data.qpos[qpos_adr] = finger_open
        except KeyError:
            pass


def _maybe_geom_id(model: mujoco.MjModel, name: str) -> Optional[int]:
    try:
        return int(model.geom(name).id)
    except KeyError:
        return None


def _maybe_site_id(model: mujoco.MjModel, name: str) -> Optional[int]:
    try:
        return int(model.site(name).id)
    except KeyError:
        return None


def collect_visibility_ids(model: mujoco.MjModel, mocap_body_name: str) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return geom/site ids for end-effector axes and mocap/target visuals."""
    ee_geom_ids = [gid for gid in (_maybe_geom_id(model, n) for n in ["ee_axis_x", "ee_axis_y", "ee_axis_z"]) if gid is not None]
    ee_site_ids = [sid for sid in [_maybe_site_id(model, SITE_NAME)] if sid is not None]

    mocap_geom_ids: list[int] = []
    mocap_site_ids: list[int] = []
    try:
        target_body_id = int(model.body(mocap_body_name).id)
        mocap_geom_ids = [int(i) for i in range(model.ngeom) if int(model.geom_bodyid[i]) == target_body_id]
        mocap_site_ids = [int(i) for i in range(model.nsite) if int(model.site_bodyid[i]) == target_body_id]
    except KeyError:
        pass
    return ee_geom_ids, ee_site_ids, mocap_geom_ids, mocap_site_ids


def apply_visual_visibility(
    model: mujoco.MjModel,
    geom_ids: list[int],
    site_ids: list[int],
    visible: bool,
    original_geom_rgba: np.ndarray,
    original_site_rgba: np.ndarray,
) -> None:
    for gid in geom_ids:
        model.geom_rgba[gid] = original_geom_rgba[gid]
        if not visible:
            model.geom_rgba[gid, 3] = 0.0
    for sid in site_ids:
        model.site_rgba[sid] = original_site_rgba[sid]
        if not visible:
            model.site_rgba[sid, 3] = 0.0


def print_terminal_visibility_help() -> None:
    print(
        "\n[终端显示/隐藏命令]\n"
        "  ee_axes on      显示机械臂末端坐标轴\n"
        "  ee_axes off     隐藏机械臂末端坐标轴\n"
        "  mocap on        显示 mocap 小方块和 mocap 坐标轴\n"
        "  mocap off       隐藏 mocap 小方块和 mocap 坐标轴\n"
        "  vis             查看当前显示状态\n"
        "  help            再次显示这些命令\n",
        flush=True,
    )


def start_terminal_visibility_thread(shared: SharedControlState) -> None:
    def loop() -> None:
        print_terminal_visibility_help()
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if line == "":
                time.sleep(0.1)
                continue
            cmd = line.strip().lower().replace("-", "_")
            if not cmd:
                continue
            with shared.lock:
                if cmd in {"ee_axes on", "ee on", "show ee", "show_ee_axes", "1 on"}:
                    shared.show_ee_axes = True
                    shared.status = "终端：显示机械臂末端坐标轴"
                    print("[terminal] 已显示机械臂末端坐标轴", flush=True)
                elif cmd in {"ee_axes off", "ee off", "hide ee", "hide_ee_axes", "1 off"}:
                    shared.show_ee_axes = False
                    shared.status = "终端：隐藏机械臂末端坐标轴"
                    print("[terminal] 已隐藏机械臂末端坐标轴", flush=True)
                elif cmd in {"mocap on", "target on", "show mocap", "show_mocap", "2 on"}:
                    shared.show_mocap = True
                    shared.status = "终端：显示 mocap/目标坐标轴"
                    print("[terminal] 已显示 mocap 小方块和坐标轴", flush=True)
                elif cmd in {"mocap off", "target off", "hide mocap", "hide_mocap", "2 off"}:
                    shared.show_mocap = False
                    shared.status = "终端：隐藏 mocap/目标坐标轴"
                    print("[terminal] 已隐藏 mocap 小方块和坐标轴", flush=True)
                elif cmd == "vis":
                    print(
                        f"[terminal] 末端坐标轴={'on' if shared.show_ee_axes else 'off'}, "
                        f"mocap={'on' if shared.show_mocap else 'off'}",
                        flush=True,
                    )
                elif cmd == "help":
                    print_terminal_visibility_help()
                else:
                    print("[terminal] 未识别命令；输入 help 查看可用命令。", flush=True)

    threading.Thread(target=loop, daemon=True).start()


def launch_ui_subprocess(script_path: Path, port: int, ui_python: str) -> subprocess.Popen[Any]:
    python_exe = ui_python
    if python_exe == "python":
        python_exe = shutil.which("python") or shutil.which("python3") or "python"
    cmd = [python_exe, str(script_path), "--ui-client", "--port", str(port)]
    print(f"[UI] 启动 PySide6 子进程：{' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def run_controller(args: argparse.Namespace) -> int:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    xml_path = Path(args.xml)
    print(f"[1/8] 使用 XML: {xml_path}", flush=True)
    if not xml_path.exists():
        print(f"找不到 XML 文件：{xml_path}", file=sys.stderr)
        return 1

    shared = SharedControlState()
    shared.gripper_value = float(args.gripper_open)
    shared.show_ee_axes = not bool(args.hide_ee_axes)
    shared.show_mocap = not bool(args.hide_mocap)

    print(f"[2/8] 启动本地控制服务：http://127.0.0.1:{args.port}", flush=True)
    server = start_http_server(shared, "127.0.0.1", int(args.port))
    if not args.no_terminal_commands:
        start_terminal_visibility_thread(shared)

    ui_proc: Optional[subprocess.Popen[Any]] = None
    if not args.no_ui:
        print("[3/8] 启动 PySide6 UI 子进程，不在 mjpython 进程里创建 QApplication。", flush=True)
        ui_proc = launch_ui_subprocess(Path(__file__).resolve(), int(args.port), args.ui_python)
    else:
        print("[3/8] 已禁用 PySide6 UI。", flush=True)

    print("[4/8] 加载 MuJoCo 模型...", flush=True)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    original_geom_rgba = model.geom_rgba.copy()
    original_site_rgba = model.site_rgba.copy()
    ee_geom_ids, ee_site_ids, mocap_geom_ids, mocap_site_ids = collect_visibility_ids(model, MOCAP_NAME)

    model.body_gravcomp[:] = float(not args.no_gravity_comp)
    model.opt.timestep = dt

    print("[5/8] 绑定 FR3 joints / actuators / site / target...", flush=True)
    site_id = model.site(SITE_NAME).id
    mocap_id = model.body(MOCAP_NAME).mocapid[0]

    joint_ids = np.array([model.joint(name).id for name in ARM_JOINT_NAMES], dtype=int)
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)
    actuator_ids = np.array([model.actuator(name).id for name in ARM_ACTUATOR_NAMES], dtype=int)
    gripper_actuator_id = find_gripper_actuator_id(model)

    key_id = model.key(KEY_NAME).id
    q0_full = model.key(KEY_NAME).qpos.copy()
    q0_arm = q0_full[qpos_ids].copy()
    arm_ranges = model.jnt_range[joint_ids].copy()

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    diag = damping * np.eye(6)
    eye7 = np.eye(7)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    print("[6/8] 打开 MuJoCo viewer...", flush=True)
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=bool(args.show_mujoco_ui),
        show_right_ui=bool(args.show_mujoco_ui),
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        set_finger_qpos_from_ctrl(model, data, shared.gripper_value)
        mujoco.mj_forward(model, data)

        # Start with target at current end-effector pose to avoid sudden motion.
        data.mocap_pos[mocap_id] = data.site(site_id).xpos.copy()
        mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(site_id).xmat)

        if gripper_actuator_id is not None:
            data.ctrl[gripper_actuator_id] = shared.gripper_value

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        # 默认不使用 MuJoCo 的全局 site-frame glyph，避免全部 site frame 过小且无法单独隐藏。
        # 末端轴和 mocap 轴由 XML 中命名的 capsule geom 显示，并可由 UI/终端单独开关。
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE if args.show_site_frame_glyphs else mujoco.mjtFrame.mjFRAME_NONE
        try:
            model.vis.scale.framelength = float(args.frame_length)
            model.vis.scale.framewidth = float(args.frame_width)
        except AttributeError:
            pass

        print("[7/8] 控制器已启动。PySide6 控制面板和 MuJoCo viewer 应该都已打开。", flush=True)
        print("[8/8] 如果没有看到 PySide6 窗口，请 Command+Tab 查找“FR3 PySide6 控制面板”。", flush=True)

        while viewer.is_running():
            step_start = time.time()

            current_pose = pose_from_site(data, site_id)

            with shared.lock:
                if shared.stop_requested:
                    data.mocap_pos[mocap_id] = data.site(site_id).xpos.copy()
                    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(site_id).xmat)
                    shared.stop_requested = False
                    shared.active = False
                    shared.active_target = None
                    shared.target_queue = []
                    shared.status = "已停止，target 已同步到当前末端"

                active_target = None if shared.active_target is None else shared.active_target.copy()
                gripper_value = float(shared.gripper_value)
                show_ee_axes = bool(shared.show_ee_axes)
                show_mocap = bool(shared.show_mocap)

            apply_visual_visibility(model, ee_geom_ids, ee_site_ids, show_ee_axes, original_geom_rgba, original_site_rgba)
            apply_visual_visibility(model, mocap_geom_ids, mocap_site_ids, show_mocap, original_geom_rgba, original_site_rgba)

            # Update target mocap from active command.
            if active_target is not None:
                data.mocap_pos[mocap_id] = active_target[:3]
                data.mocap_quat[mocap_id] = rpy_to_quat(*active_target[3:])

            if gripper_actuator_id is not None:
                data.ctrl[gripper_actuator_id] = gripper_value

            # Spatial velocity / twist.
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Site Jacobian restricted to the 7 controlled arm joints.
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = np.vstack((jacp[:, dof_ids], jacr[:, dof_ids]))

            # Damped least squares pseudo-inverse.
            j_damped = jac.T @ np.linalg.solve(jac @ jac.T + diag, np.eye(6))
            dq = j_damped @ twist

            # Nullspace home bias.
            q_arm = data.qpos[qpos_ids].copy()
            nullspace = eye7 - j_damped @ jac
            dq += nullspace @ (Kn * (q0_arm - q_arm))

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            # Integrate arm joint velocities into target joint positions.
            q = data.qpos.copy()
            v = np.zeros(model.nv)
            v[dof_ids] = dq
            mujoco.mj_integratePos(model, q, v, integration_dt)
            q[qpos_ids] = np.clip(q[qpos_ids], arm_ranges[:, 0], arm_ranges[:, 1])

            # Position servo commands for arm joints only.
            data.ctrl[actuator_ids] = q[qpos_ids]

            mujoco.mj_step(model, data)

            # Waypoint completion check.
            pos_err, ori_err = 0.0, 0.0
            with shared.lock:
                if shared.active_target is not None:
                    pos_err, ori_err = pose_error_for_reached(model, data, site_id, shared.active_target)
                    if pos_err < args.position_tolerance and ori_err < args.orientation_tolerance:
                        if shared.target_queue:
                            completed_count = 1  # status string only; exact count is not critical.
                            shared.active_target = shared.target_queue.pop(0)
                            total_left = len(shared.target_queue) + 1
                            shared.status = f"已到达 waypoint，继续下一个；剩余 {total_left} 个"
                        else:
                            shared.active_target = None
                            shared.active = False
                            shared.status = "目标已到达"
                shared.current_pose = pose_from_site(data, site_id)
                shared.pos_err = pos_err
                shared.ori_err = ori_err

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("MuJoCo viewer 已关闭，正在退出...", flush=True)
    server.shutdown()
    if ui_proc is not None and ui_proc.poll() is None:
        ui_proc.terminate()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FR3 MuJoCo controller with a PySide6 UI subprocess")
    parser.add_argument("--xml", default="franka_fr3/scene_fr3_with_gripper_full_close_visual_toggle.xml")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ui-python", default="python", help="Python executable used to run the PySide6 UI subprocess")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--ui-client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-gravity-comp", action="store_true")
    parser.add_argument("--show-mujoco-ui", action="store_true")
    parser.add_argument("--gripper-open", type=float, default=0.0)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance", type=float, default=0.035)
    parser.add_argument("--frame-length", type=float, default=0.09, help="MuJoCo site-frame glyph length, only used with --show-site-frame-glyphs")
    parser.add_argument("--frame-width", type=float, default=0.003, help="MuJoCo site-frame glyph width, only used with --show-site-frame-glyphs")
    parser.add_argument("--hide-ee-axes", action="store_true", help="Start with end-effector axes hidden")
    parser.add_argument("--hide-mocap", action="store_true", help="Start with mocap target and axes hidden")
    parser.add_argument("--show-site-frame-glyphs", action="store_true", help="Also show MuJoCo's built-in site frame glyphs")
    parser.add_argument("--no-terminal-commands", action="store_true", help="Disable terminal visibility commands")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.ui_client:
        return run_ui_client(int(args.port))
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
