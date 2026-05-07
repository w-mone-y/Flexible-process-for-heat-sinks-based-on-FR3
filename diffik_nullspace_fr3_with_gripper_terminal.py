"""Differential IK + nullspace controller for Franka Research 3 in MuJoCo.

This version keeps the FR3 + Panda gripper setup and adds a terminal command
interface:

  show
      Print the current end-effector 6D pose: x y z roll pitch yaw.

  move_one_step
      Enter one 6D pose. The mocap target is moved there, and the arm follows.

  move_multi_step
      Enter multiple 6D poses, one per line. Type "end" to execute them in order.

Pose convention:
  x y z roll pitch yaw
  - position is in meters, in the MuJoCo world frame
  - roll/pitch/yaw are ZYX RPY angles by default, in radians
  - you may enter "deg x y z roll pitch yaw" for a single line in degrees
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain. One value per controlled arm joint.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=float)

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785


@dataclass(frozen=True)
class RobotConfig:
    xml_path: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    site_name: str = "attachment_site"
    key_name: str = "home"
    mocap_name: str = "target"
    gripper_actuator_name: str | None = "gripper"


@dataclass(frozen=True)
class PoseCommand:
    position: np.ndarray  # shape: (3,)
    quat: np.ndarray      # MuJoCo quaternion order: w x y z
    raw_pose: np.ndarray  # x y z roll pitch yaw, radians


ROBOT_CONFIGS: dict[str, RobotConfig] = {
    "fr3": RobotConfig(
        xml_path="franka_fr3/scene_fr3_with_gripper_full_close.xml",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Differential IK controller for Panda/FR3 in MuJoCo.")
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_CONFIGS),
        default="fr3",
        help="Robot preset to use. Default: fr3.",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Optional scene XML path. Overrides the preset path.",
    )
    parser.add_argument(
        "--no-gravity-comp",
        action="store_true",
        help="Disable MuJoCo body gravity compensation.",
    )
    parser.add_argument(
        "--hide-ui",
        action="store_true",
        help="Hide MuJoCo side UI. By default the UI is shown so the gripper control slider is available.",
    )
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=255.0,
        help="Initial gripper control value. 0 is closed, 255 is fully open for this XML. Default: 255.",
    )
    parser.add_argument(
        "--angle-unit",
        choices=("rad", "deg"),
        default="rad",
        help="Default unit for roll/pitch/yaw typed in the terminal. Default: rad.",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.005,
        help="Waypoint reached tolerance for position, in meters. Default: 0.005.",
    )
    parser.add_argument(
        "--orientation-tolerance",
        type=float,
        default=0.035,
        help="Waypoint reached tolerance for orientation, in radians. Default: 0.035.",
    )
    return parser.parse_args()


def get_joint_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return joint ids, qpos addresses, and dof addresses for 1-DoF joints."""
    joint_ids: list[int] = []
    qpos_ids: list[int] = []
    dof_ids: list[int] = []

    for name in joint_names:
        jid = model.joint(name).id
        joint_ids.append(jid)
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        dof_ids.append(int(model.jnt_dofadr[jid]))

    return np.asarray(joint_ids), np.asarray(qpos_ids), np.asarray(dof_ids)


def get_actuator_ids(model: mujoco.MjModel, actuator_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([model.actuator(name).id for name in actuator_names], dtype=int)


def initialize_gripper_from_slider_value(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_actuator_id: int | None,
    gripper_value: float,
) -> None:
    """Set both gripper control and initial finger qpos."""
    if gripper_actuator_id is None:
        return

    ctrl_lo, ctrl_hi = model.actuator_ctrlrange[gripper_actuator_id]
    ctrl = float(np.clip(gripper_value, ctrl_lo, ctrl_hi))
    data.ctrl[gripper_actuator_id] = ctrl

    alpha = 0.0 if ctrl_hi == ctrl_lo else (ctrl - ctrl_lo) / (ctrl_hi - ctrl_lo)
    for finger_name in ("finger_joint1", "finger_joint2"):
        try:
            jid = model.joint(finger_name).id
        except KeyError:
            continue
        qpos_id = int(model.jnt_qposadr[jid])
        if model.jnt_limited[jid]:
            q_lo, q_hi = model.jnt_range[jid]
            data.qpos[qpos_id] = q_lo + alpha * (q_hi - q_lo)

    mujoco.mj_forward(model, data)


def clip_controlled_joints(model: mujoco.MjModel, q: np.ndarray, joint_ids: np.ndarray, qpos_ids: np.ndarray) -> None:
    """Clip only the controlled hinge joints to their model-defined ranges."""
    for jid, qpos_id in zip(joint_ids, qpos_ids, strict=True):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q[qpos_id] = np.clip(q[qpos_id], lo, hi)


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
        # Gimbal-lock fallback.
        roll = math.atan2(-r[1, 2], r[1, 1])
        yaw = 0.0

    return np.asarray([roll, pitch, yaw], dtype=float)


def parse_pose_line(line: str, default_angle_unit: str) -> PoseCommand:
    """Parse 'x y z roll pitch yaw' or 'deg x y z roll pitch yaw'."""
    cleaned = line.replace(",", " ").strip()
    tokens = cleaned.split()
    if not tokens:
        raise ValueError("empty input")

    unit = default_angle_unit
    if tokens[0].lower() in {"rad", "deg"}:
        unit = tokens[0].lower()
        tokens = tokens[1:]

    if len(tokens) != 6:
        raise ValueError("expected 6 numbers: x y z roll pitch yaw")

    values = np.asarray([float(x) for x in tokens], dtype=float)
    if unit == "deg":
        values[3:] = np.deg2rad(values[3:])

    quat = quat_from_rpy(float(values[3]), float(values[4]), float(values[5]))
    return PoseCommand(position=values[:3].copy(), quat=quat, raw_pose=values.copy())


def format_pose(values: np.ndarray) -> str:
    return " ".join(f"{x: .6f}" for x in values)


def start_terminal_thread(command_queue: queue.Queue, default_angle_unit: str) -> threading.Thread:
    """Start a daemon thread that reads terminal commands without blocking the viewer."""

    def prompt_loop() -> None:
        print("\n终端命令已开启：show | move_one_step | move_multi_step | quit", flush=True)
        print("位姿格式：x y z roll pitch yaw。默认角度单位为 " + default_angle_unit + "；也可输入：deg x y z roll pitch yaw", flush=True)
        while True:
            try:
                command = input("\n请输入命令 [show/move_one_step/move_multi_step/quit]: ").strip().lower()
            except EOFError:
                command_queue.put({"type": "quit"})
                break
            except KeyboardInterrupt:
                command_queue.put({"type": "quit"})
                break

            if command == "show":
                command_queue.put({"type": "show"})
            elif command == "move_one_step":
                try:
                    line = input("请输入一个六维位姿: ").strip()
                    pose = parse_pose_line(line, default_angle_unit)
                except Exception as exc:
                    print(f"输入无效：{exc}", flush=True)
                    continue
                command_queue.put({"type": "move", "waypoints": [pose], "mode": "move_one_step"})
            elif command == "move_multi_step":
                poses: list[PoseCommand] = []
                print("逐行输入六维位姿；输入 end 后开始执行。", flush=True)
                while True:
                    try:
                        line = input(f"waypoint {len(poses) + 1}> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        line = "end"
                    if line.lower() == "end":
                        break
                    if not line:
                        continue
                    try:
                        poses.append(parse_pose_line(line, default_angle_unit))
                    except Exception as exc:
                        print(f"这一行无效，已忽略：{exc}", flush=True)
                if poses:
                    command_queue.put({"type": "move", "waypoints": poses, "mode": "move_multi_step"})
                else:
                    print("没有收到有效 waypoint，已取消。", flush=True)
            elif command in {"quit", "exit", "q"}:
                command_queue.put({"type": "quit"})
                break
            elif command in {"help", "h", "?"}:
                print("可用命令：show | move_one_step | move_multi_step | quit", flush=True)
            elif not command:
                continue
            else:
                print("未知命令。请输入 show、move_one_step、move_multi_step 或 quit。", flush=True)

    thread = threading.Thread(target=prompt_loop, daemon=True)
    thread.start()
    return thread


def set_mocap_pose(data: mujoco.MjData, mocap_id: int, pose: PoseCommand) -> None:
    data.mocap_pos[mocap_id] = pose.position
    data.mocap_quat[mocap_id] = pose.quat


def current_site_pose_6d(data: mujoco.MjData, site_id: int) -> np.ndarray:
    position = np.array(data.site(site_id).xpos, copy=True)
    rpy = rpy_from_mat(data.site(site_id).xmat)
    return np.concatenate([position, rpy])


def orientation_error_norm(data: mujoco.MjData, site_id: int, mocap_id: int) -> float:
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    omega = np.zeros(3)
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(omega, error_quat, 1.0)
    return float(np.linalg.norm(omega))


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    args = parse_args()
    cfg = ROBOT_CONFIGS[args.robot]
    xml_path = Path(args.xml or cfg.xml_path)

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Enable/disable gravity compensation.
    model.body_gravcomp[:] = 0.0 if args.no_gravity_comp else float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_id = model.site(cfg.site_name).id

    # Controlled joint/actuator ids.
    joint_ids, qpos_ids, dof_ids = get_joint_addresses(model, cfg.joint_names)
    actuator_ids = get_actuator_ids(model, cfg.actuator_names)
    gripper_actuator_id = None
    if cfg.gripper_actuator_name is not None:
        try:
            gripper_actuator_id = model.actuator(cfg.gripper_actuator_name).id
        except KeyError:
            gripper_actuator_id = None

    if len(qpos_ids) != len(Kn):
        raise ValueError(f"Kn has length {len(Kn)}, but {len(qpos_ids)} joints are controlled.")

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key(cfg.key_name).id
    q0_full = np.array(model.key(cfg.key_name).qpos, copy=True)
    q0_arm = q0_full[qpos_ids]

    # Mocap body we will control with the mouse or terminal commands.
    mocap_id = model.body(cfg.mocap_name).mocapid[0]
    if mocap_id < 0:
        raise ValueError(f"Body '{cfg.mocap_name}' exists but is not a mocap body.")

    # Pre-allocate numpy arrays.
    jac_full = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye_arm = np.eye(len(dof_ids))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    command_queue: queue.Queue = queue.Queue()
    current_goal: PoseCommand | None = None
    remaining_goals: list[PoseCommand] = []
    active_mode = ""
    active_goal_index = 0
    total_goal_count = 0
    should_quit = False

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=not args.hide_ui,
        show_right_ui=not args.hide_ui,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the gripper actuator and the finger joint qpos. The arm
        # controller below intentionally does not write to this actuator, so the
        # MuJoCo control slider can adjust it after launch.
        initialize_gripper_from_slider_value(model, data, gripper_actuator_id, args.gripper_open)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Make sure the visual groups used by the target and ee-axis markers are visible.
        # group 1: custom axis geoms; group 4: FR3 default sites.
        for group_id in (1, 4):
            try:
                viewer.opt.geomgroup[group_id] = 1
            except Exception:
                pass
            try:
                viewer.opt.sitegroup[group_id] = 1
            except Exception:
                pass

        # Native frame glyph scale. Keep it small because your XML already has custom axes.
        try:
            model.vis.scale.framelength = 0.03
            model.vis.scale.framewidth = 0.0012
        except Exception:
            pass

        start_terminal_thread(command_queue, args.angle_unit)

        while viewer.is_running() and not should_quit:
            step_start = time.time()

            # Process terminal commands in the MuJoCo/main thread.
            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break

                if command["type"] == "show":
                    pose = current_site_pose_6d(data, site_id)
                    print("\n当前末端六维位姿 [x y z roll pitch yaw]，角度单位 rad：", flush=True)
                    print(format_pose(pose), flush=True)
                    print("对应角度 deg：" + format_pose(np.concatenate([pose[:3], np.rad2deg(pose[3:])])), flush=True)
                elif command["type"] == "move":
                    waypoints: list[PoseCommand] = list(command["waypoints"])
                    if not waypoints:
                        continue
                    active_mode = str(command.get("mode", "move"))
                    remaining_goals = waypoints[1:]
                    current_goal = waypoints[0]
                    active_goal_index = 1
                    total_goal_count = len(waypoints)
                    set_mocap_pose(data, mocap_id, current_goal)
                    print(
                        f"\n开始执行 {active_mode}: waypoint {active_goal_index}/{total_goal_count} -> "
                        f"{format_pose(current_goal.raw_pose)}",
                        flush=True,
                    )
                elif command["type"] == "quit":
                    print("收到 quit，正在退出 viewer...", flush=True)
                    should_quit = True
                    break

            # Advance waypoint sequence when current target is reached.
            if current_goal is not None:
                pos_err = float(np.linalg.norm(data.mocap_pos[mocap_id] - data.site(site_id).xpos))
                ori_err = orientation_error_norm(data, site_id, mocap_id)
                if pos_err < args.position_tolerance and ori_err < args.orientation_tolerance:
                    print(
                        f"到达 waypoint {active_goal_index}/{total_goal_count} "
                        f"(pos_err={pos_err:.4f} m, ori_err={ori_err:.4f} rad)",
                        flush=True,
                    )
                    if remaining_goals:
                        current_goal = remaining_goals.pop(0)
                        active_goal_index += 1
                        set_mocap_pose(data, mocap_id, current_goal)
                        print(
                            f"继续执行 waypoint {active_goal_index}/{total_goal_count} -> "
                            f"{format_pose(current_goal.raw_pose)}",
                            flush=True,
                        )
                    else:
                        print(f"{active_mode} 执行完成。", flush=True)
                        current_goal = None
                        active_mode = ""
                        active_goal_index = 0
                        total_goal_count = 0

            # Spatial velocity aka twist.
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt

            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Site Jacobian. Slice to the 7 controlled arm DoFs.
            mujoco.mj_jacSite(model, data, jac_full[:3], jac_full[3:], site_id)
            jac = jac_full[:, dof_ids]

            # Damped least-squares inverse. Shape: 7 x 6.
            jac_dls = jac.T @ np.linalg.solve(jac @ jac.T + diag, np.eye(6))

            # Main differential IK task.
            dq_arm = jac_dls @ twist

            # Nullspace control biasing joint velocities towards the home configuration.
            q_arm = data.qpos[qpos_ids]
            nullspace_projector = eye_arm - jac_dls @ jac
            dq_arm += nullspace_projector @ (Kn * (q0_arm - q_arm))

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq_arm).max()
            if dq_abs_max > max_angvel:
                dq_arm *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint position commands.
            q = data.qpos.copy()
            dq_full = np.zeros(model.nv)
            dq_full[dof_ids] = dq_arm
            mujoco.mj_integratePos(model, q, dq_full, integration_dt)
            clip_controlled_joints(model, q, joint_ids, qpos_ids)

            # Set only the 7 arm position controls. Do not write the gripper
            # actuator here; leave it to the MuJoCo control slider.
            data.ctrl[actuator_ids] = q[qpos_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
