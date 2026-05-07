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
  * the MuJoCo control UI is shown so the gripper actuator can be adjusted with a slider
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
    gripper_actuator_name: str | None = "gripper"


ROBOT_CONFIGS: dict[str, RobotConfig] = {
    # This scene is generated from your FR3 scene with a mocap target added.
    "fr3": RobotConfig(
        xml_path="franka_fr3/scene_fr3_with_gripper_full_close.xml",
        joint_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
        actuator_names=tuple(f"fr3_joint{i}" for i in range(1, 8)),
    ),
    # Kept for comparison/debugging. Depending on your Panda XML, actuator names may
    # be joint1..joint7 or actuator1..actuator7; override with --xml or edit here.
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



def initialize_gripper_from_slider_value(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_actuator_id: int | None,
    gripper_value: float,
) -> None:
    """Set both gripper control and initial finger qpos.

    The gripper actuator uses the Panda-style 0..255 control range, while each
    finger joint uses 0..0.04 m.  Setting qpos here prevents a launch with
    --gripper-open 0 from visually starting at the XML keyframe's open qpos.
    """
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

        # Enable site frame visualization.  This draws MuJoCo's native coordinate
        # frame at every site, including attachment_site.  The extra geom axes in
        # fr3_with_ee_axes.xml make the end-effector frame obvious even when the
        # native site frame is visually small.
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

        # Enlarge MuJoCo's native frame glyphs if this MuJoCo version exposes the scale fields.
        try:
            model.vis.scale.framelength = 0.045
            model.vis.scale.framewidth = 0.002
        except Exception:
            pass

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
