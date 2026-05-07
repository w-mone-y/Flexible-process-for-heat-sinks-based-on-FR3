"""Differential IK + nullspace controller for Franka Research 3 in MuJoCo.

This is a migrated version of the Panda controller.  The control law is still:
  end-effector pose error -> 6D twist -> damped-least-squares IK -> nullspace home bias

Main migration changes:
  * model/scene path is configurable and defaults to FR3
  * FR3 joint/actuator names are used: fr3_joint1 ... fr3_joint7
  * the Jacobian is sliced to the 7 controlled arm DoFs, so the code is robust even
    if the model later contains extra DoFs such as fingers or other joints
  * the nullspace projector uses the same damped inverse as the main task
  * joint-position integration updates only the controlled arm joints
"""

from __future__ import annotations

import argparse
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


ROBOT_CONFIGS: dict[str, RobotConfig] = {
    # This scene is generated from your FR3 scene with a mocap target added.
    "fr3": RobotConfig(
        xml_path="scene_fr3_with_target.xml",
        joint_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
        actuator_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
    ),
    # Kept for comparison/debugging. Depending on your Panda XML, actuator names may
    # be joint1..joint7 or actuator1..actuator7; override with --xml or edit here.
    "panda": RobotConfig(
        xml_path="franka_emika_panda/scene.xml",
        joint_names=tuple(f"joint{i}" for i in range(1, 8)),
        actuator_names=tuple(f"joint{i}" for i in range(1, 8)),
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
    return parser.parse_args()


def get_joint_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return joint ids, qpos addresses, and dof addresses for 1-DoF joints."""
    joint_ids: list[int] = []
    qpos_ids: list[int] = []
    dof_ids: list[int] = []

    for name in joint_names:
        jid = model.joint(name).id
        joint_ids.append(jid)

        # FR3/Panda arm joints are hinge joints, so each has one qpos and one dof.
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        dof_ids.append(int(model.jnt_dofadr[jid]))

    return np.asarray(joint_ids), np.asarray(qpos_ids), np.asarray(dof_ids)


def get_actuator_ids(model: mujoco.MjModel, actuator_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([model.actuator(name).id for name in actuator_names], dtype=int)


def clip_controlled_joints(model: mujoco.MjModel, q: np.ndarray, joint_ids: np.ndarray, qpos_ids: np.ndarray) -> None:
    """Clip only the controlled hinge joints to their model-defined ranges."""
    for jid, qpos_id in zip(joint_ids, qpos_ids, strict=True):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q[qpos_id] = np.clip(q[qpos_id], lo, hi)


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

    if len(qpos_ids) != len(Kn):
        raise ValueError(f"Kn has length {len(Kn)}, but {len(qpos_ids)} joints are controlled.")

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key(cfg.key_name).id
    q0_full = np.array(model.key(cfg.key_name).qpos, copy=True)
    q0_arm = q0_full[qpos_ids]

    # Mocap body we will control with our mouse.
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

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

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

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[qpos_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
