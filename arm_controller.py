"""Single-arm controller for one prefixed FR3: 5D IK + deadband hold +
wrist_spin decoupling + persistent setpoint + skill-step executor + parking.

Tool support: `tool_offsets` maps a TCP name to its fixed offset in the
EE-site frame. For arms with permanently mounted tools it can be filled at
init; for Arm2 the offsets are (re)written by Arm2ToolManager whenever a
quick-change tool is docked or returned.
"""
from __future__ import annotations

import argparse
import math
from typing import Any, Callable, Optional

import numpy as np

from skill_library import (
    GRIPPER_OPEN_CTRL,
    SCREW_SPIN_SPEED_DEG_S,
    SETPOINT_TRACKING_BAND,
    PoseCommand,
    SkillStep,
    mat_from_quat,
    mat_to_quat,
    normalize_vec,
    pose_cmd,
    quat_mul,
    quat_rot_z,
    rot_z_mat,
)
from object_manager import COMPONENT_INSTANCES


class ArmController:
    def __init__(self, model: Any, data: Any, name: str, args: argparse.Namespace) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.name = name
        self.args = args
        p = f"{name}_"

        self.site_id = int(model.site(p + "attachment_site").id)
        self.mocap_id = int(model.body(p + "target").mocapid[0])
        self.hand_body_id = int(model.body(p + "hand").id)
        joint_ids = np.asarray([int(model.joint(f"{p}fr3_joint{i}").id) for i in range(1, 8)], dtype=int)
        self.qpos_ids = np.asarray([int(model.jnt_qposadr[j]) for j in joint_ids], dtype=int)
        self.dof_ids = np.asarray([int(model.jnt_dofadr[j]) for j in joint_ids], dtype=int)
        self.actuator_ids = np.asarray([int(model.actuator(f"{p}fr3_joint{i}").id) for i in range(1, 8)], dtype=int)
        try:
            self.gripper_act_id: Optional[int] = int(model.actuator(p + "gripper").id)
        except KeyError:
            # Arm3 carries a directly mounted inspection camera and therefore
            # has no hand, finger joints, tendon, or gripper actuator.
            self.gripper_act_id = None
        spin_joint_id = int(model.joint(p + "wrist_spin").id)
        self.spin_qpos_id = int(model.jnt_qposadr[spin_joint_id])
        self.spin_act_id = int(model.actuator(p + "wrist_spin").id)
        self.weld_ids: dict[str, int] = {}
        for comp in COMPONENT_INSTANCES:
            try:
                self.weld_ids[comp] = int(model.equality(f"{name}_grasp_{comp}").id)
            except KeyError:
                pass

        self.joint_lower = np.full(7, -np.inf)
        self.joint_upper = np.full(7, np.inf)
        self.joint_mid = np.zeros(7)
        self.joint_half_range = np.ones(7)
        for i, jid in enumerate(joint_ids):
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                self.joint_lower[i] = float(lo)
                self.joint_upper[i] = float(hi)
                self.joint_mid[i] = 0.5 * (float(lo) + float(hi))
                self.joint_half_range[i] = max(0.5 * (float(hi) - float(lo)), 1e-6)

        self.jac_full = np.zeros((6, model.nv), dtype=float)
        self.eye_arm = np.eye(7)
        self.q_ctrl_target = np.array(data.qpos[self.qpos_ids], copy=True)
        self.arm_in_deadband = False
        self.wrist_spin_target_rad = float(data.qpos[self.spin_qpos_id])
        self.gripper_ctrl = GRIPPER_OPEN_CTRL

        # Home (parking) pose from the keyframe state: idle arms return here so
        # they never hover over shared areas and block other arms.
        home_quat = np.zeros(4, dtype=float)
        self.mujoco.mju_mat2Quat(home_quat, data.site(self.site_id).xmat)
        self.home_pose = pose_cmd(np.array(data.site(self.site_id).xpos, copy=True), home_quat)

        # TCP offsets in the EE-site frame. Empty until a tool provides them:
        # for Arm2, Arm2ToolManager rewrites this dict on every dock/undock.
        self.tool_offsets: dict[str, np.ndarray] = {}

        # Skill executor state.
        self.steps: list[SkillStep] = []
        self.step_idx = 0
        self.phase = "idle"          # idle | move | spin | dwell
        self.dwell_until = 0.0
        self.spin_remaining = 0.0
        self.spin_center: Optional[np.ndarray] = None
        self.held_component = ""
        self.status = "idle"
        self.manual_hold = False     # user-commanded pose: suppress auto-parking

    # -- mocap helpers -----------------------------------------------------
    def set_mocap_pose(self, cmd: PoseCommand) -> None:
        self.data.mocap_pos[self.mocap_id] = cmd.position
        self.data.mocap_quat[self.mocap_id] = cmd.quat

    def snap_mocap_to_site(self) -> None:
        quat = np.zeros(4, dtype=float)
        self.mujoco.mju_mat2Quat(quat, self.data.site(self.site_id).xmat)
        self.data.mocap_pos[self.mocap_id] = self.data.site(self.site_id).xpos
        self.data.mocap_quat[self.mocap_id] = quat

    def reset_hold(self) -> None:
        self.q_ctrl_target = np.array(self.data.qpos[self.qpos_ids], copy=True)
        self.wrist_spin_target_rad = float(self.data.qpos[self.spin_qpos_id])
        self.snap_mocap_to_site()

    def pos_error(self) -> float:
        return float(np.linalg.norm(self.data.mocap_pos[self.mocap_id] - self.data.site(self.site_id).xpos))

    def ori_error(self) -> float:
        site_quat = np.zeros(4, dtype=float)
        site_quat_conj = np.zeros(4, dtype=float)
        error_quat = np.zeros(4, dtype=float)
        omega = np.zeros(3, dtype=float)
        self.mujoco.mju_mat2Quat(site_quat, self.data.site(self.site_id).xmat)
        self.mujoco.mju_negQuat(site_quat_conj, site_quat)
        self.mujoco.mju_mulQuat(error_quat, self.data.mocap_quat[self.mocap_id], site_quat_conj)
        self.mujoco.mju_quat2Vel(omega, error_quat, 1.0)
        return float(np.linalg.norm(omega))

    # -- weld attach/detach --------------------------------------------------
    def attach_component(self, instance: str) -> dict[str, np.ndarray]:
        body_id = int(self.model.body(instance).id)
        eq_id = self.weld_ids[instance]
        hand_pos = np.asarray(self.data.body(self.hand_body_id).xpos, dtype=float)
        hand_mat = np.asarray(self.data.body(self.hand_body_id).xmat, dtype=float).reshape(3, 3)
        obj_pos = np.asarray(self.data.body(body_id).xpos, dtype=float)
        obj_mat = np.asarray(self.data.body(body_id).xmat, dtype=float).reshape(3, 3)
        rel_pos = hand_mat.T @ (obj_pos - hand_pos)
        rel_quat = mat_to_quat(hand_mat.T @ obj_mat)
        self.model.eq_data[eq_id, :] = 0.0
        self.model.eq_data[eq_id, 3:6] = rel_pos
        self.model.eq_data[eq_id, 6:10] = rel_quat
        self.model.eq_data[eq_id, 10] = 1.0
        self.data.eq_active[eq_id] = 1
        site_pos = np.asarray(self.data.site(self.site_id).xpos, dtype=float)
        site_mat = np.asarray(self.data.site(self.site_id).xmat, dtype=float).reshape(3, 3)
        self.held_component = instance
        return {
            "pos_site_obj": site_mat.T @ (obj_pos - site_pos),
            "mat_site_obj": site_mat.T @ obj_mat,
        }

    def detach_component(self, instance: str) -> None:
        self.data.eq_active[self.weld_ids[instance]] = 0
        self.held_component = ""

    # -- skill executor ------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.phase != "idle"

    def start_steps(self, steps: list[SkillStep], status: str) -> None:
        self.steps = steps
        self.step_idx = 0
        self.phase = "move"
        self.status = status
        self.set_mocap_pose(steps[0].resolve_pose())

    def cancel(self) -> None:
        self.steps = []
        self.step_idx = 0
        self.phase = "idle"
        self.status = "idle"
        self.reset_hold()

    @property
    def parking(self) -> bool:
        return self.busy and self.status == "park"

    def dist_to_home(self) -> float:
        return float(np.linalg.norm(self.home_pose.position - self.data.site(self.site_id).xpos))

    def start_park(self) -> None:
        """Vertical lift first, then travel to the home pose. Loose tolerances:
        parking is about clearing shared space, not precision."""
        cur = np.array(self.data.site(self.site_id).xpos, copy=True)
        lift_z = max(float(cur[2]), float(self.home_pose.position[2]) - 0.10)
        lift_pos = np.asarray([cur[0], cur[1], lift_z], dtype=float)
        steps = [
            SkillStep("park_lift", pose=pose_cmd(lift_pos, self.home_pose.quat), pos_tol=0.02, ori_tol=0.3, dwell_s=0.05),
            SkillStep("park_home", pose=self.home_pose, pos_tol=0.015, ori_tol=0.2, dwell_s=0.05),
        ]
        self.start_steps(steps, "park")

    def skill_tick(self, now: float, on_action: Callable[["ArmController", SkillStep], None]) -> bool:
        """Advance the step machine. Returns True when the whole sequence ends."""
        if self.phase == "idle" or not self.steps:
            return False
        step = self.steps[self.step_idx]
        if self.phase == "move":
            if self.pos_error() < float(step.pos_tol) and self.ori_error() < float(step.ori_tol):
                if step.action:
                    on_action(self, step)
                if float(step.spin_rad) > 1e-9:
                    self.phase = "spin"
                    self.spin_remaining = float(step.spin_rad)
                    self.spin_center = None
                    if step.spin_center is not None:
                        self.spin_center = np.asarray(step.spin_center, dtype=float)
                else:
                    self.phase = "dwell"
                    self.dwell_until = now + float(step.dwell_s)
        elif self.phase == "spin":
            delta = math.radians(SCREW_SPIN_SPEED_DEG_S) * float(self.args.dt)
            delta = min(delta, self.spin_remaining)
            quat = np.asarray(self.data.mocap_quat[self.mocap_id], dtype=float)
            self.data.mocap_quat[self.mocap_id] = quat_mul(quat, quat_rot_z(-delta))
            if self.spin_center is not None:
                # Orbit compensation: rotate the EE target around the vertical
                # tool axis so an off-center tool tip stays ON that axis while
                # the wrist spins (the screwdriver bit turns in place). A body
                # -z twist of -delta equals a world-z rotation of -delta*sign(az)
                # where az is the world-z component of the EE z axis (az<0 for
                # the top-down tool pose, so the orbit runs at +delta).
                az = float(mat_from_quat(quat)[2, 2])
                world_angle = -delta * (1.0 if az >= 0.0 else -1.0)
                pos = np.asarray(self.data.mocap_pos[self.mocap_id], dtype=float)
                rel = pos[:2] - self.spin_center
                rot = rot_z_mat(world_angle)[:2, :2]
                self.data.mocap_pos[self.mocap_id][:2] = self.spin_center + rot @ rel
            self.spin_remaining -= delta
            if self.spin_remaining <= 1e-9:
                self.phase = "dwell"
                self.dwell_until = now + float(step.dwell_s)
        elif self.phase == "dwell" and now >= self.dwell_until:
            self.step_idx += 1
            if self.step_idx >= len(self.steps):
                self.steps = []
                self.step_idx = 0
                self.phase = "idle"
                self.status = "idle"
                self.snap_mocap_to_site()
                return True
            self.phase = "move"
            self.set_mocap_pose(self.steps[self.step_idx].resolve_pose())
        return False

    def current_step_label(self) -> str:
        if self.phase != "idle" and self.step_idx < len(self.steps):
            return self.steps[self.step_idx].label
        return ""

    # -- per-frame control law ------------------------------------------------
    def control_tick(self) -> None:
        args = self.args
        mujoco = self.mujoco
        data = self.data

        if self.gripper_act_id is not None:
            data.ctrl[self.gripper_act_id] = float(self.gripper_ctrl)

        dx = data.mocap_pos[self.mocap_id] - data.site(self.site_id).xpos
        target_mat = mat_from_quat(data.mocap_quat[self.mocap_id])
        site_mat = np.asarray(data.site(self.site_id).xmat, dtype=float).reshape(3, 3)
        z_cur = normalize_vec(site_mat[:, 2])
        z_target = normalize_vec(target_mat[:, 2])
        z_cross = np.cross(z_cur, z_target)
        z_cross_norm = float(np.linalg.norm(z_cross))
        z_axis_error = float(math.atan2(z_cross_norm, float(np.dot(z_cur, z_target))))
        z_axis_error_vec = (z_cross / z_cross_norm * z_axis_error) if z_cross_norm > 1e-9 else np.zeros(3)

        q_arm = np.array(data.qpos[self.qpos_ids], copy=True)
        dx_norm = float(np.linalg.norm(dx))
        if self.arm_in_deadband:
            if dx_norm > 5.0 * float(args.arm_deadband_pos) or z_axis_error > 5.0 * float(args.arm_deadband_ori):
                self.arm_in_deadband = False
        elif dx_norm < float(args.arm_deadband_pos) and z_axis_error < float(args.arm_deadband_ori):
            self.arm_in_deadband = True

        if self.arm_in_deadband:
            dq_arm = np.zeros(7)
        else:
            ref = np.asarray([1.0, 0.0, 0.0])
            if abs(float(np.dot(ref, z_cur))) > 0.85:
                ref = np.asarray([0.0, 1.0, 0.0])
            b1 = normalize_vec(ref - float(np.dot(ref, z_cur)) * z_cur)
            b2 = normalize_vec(np.cross(z_cur, b1))
            task5 = np.zeros(5)
            task5[:3] = float(args.kpos) * dx / float(args.dt)
            omega_z = float(args.kori) * z_axis_error_vec / float(args.dt)
            task5[3:] = np.asarray([float(np.dot(b1, omega_z)), float(np.dot(b2, omega_z))])

            mujoco.mj_jacSite(self.model, data, self.jac_full[:3], self.jac_full[3:], self.site_id)
            jac_pos = self.jac_full[:3, self.dof_ids]
            jac_rot = self.jac_full[3:, self.dof_ids]
            jac5 = np.vstack([jac_pos, np.vstack([b1, b2]) @ jac_rot])
            diag5 = float(args.damping) * np.eye(5)
            primary_scale = (dx_norm / max(float(args.position_tolerance), 1e-6)) + (
                z_axis_error / max(float(args.orientation_tolerance), 1e-6)
            )
            try:
                jac5_dls = jac5.T @ np.linalg.solve(jac5 @ jac5.T + diag5, np.eye(5))
                dq_arm = jac5_dls @ task5
                nullspace_projector = self.eye_arm - jac5_dls @ jac5
                if primary_scale < float(args.joint_secondary_gate):
                    gamma_secondary = 0.0
                else:
                    gamma_secondary = float(np.clip(primary_scale / max(float(args.joint_secondary_gate), 1e-6), 0.0, 1.0))
                center_velocity = float(args.joint_centering_gain) * (self.joint_mid - q_arm)
                eta = np.clip((q_arm - self.joint_mid) / np.maximum(self.joint_half_range, 1e-8), -0.98, 0.98)
                barrier_shape = -eta / np.maximum(1.0 - eta * eta, 1e-3) ** 2
                barrier_velocity = np.clip(
                    float(args.joint_limit_barrier_gain) * barrier_shape * self.joint_half_range,
                    -float(args.max_nullspace_speed),
                    float(args.max_nullspace_speed),
                )
                dq_arm += gamma_secondary * (nullspace_projector @ (center_velocity + barrier_velocity))
            except np.linalg.LinAlgError:
                dq_arm = np.zeros(7)

        # wrist_spin: close the remaining twist DOF about the tool z axis.
        current_spin = float(data.qpos[self.spin_qpos_id])
        site_no_spin_mat = site_mat @ rot_z_mat(-current_spin)
        x_cur = site_no_spin_mat[:, 0] - float(np.dot(site_no_spin_mat[:, 0], z_target)) * z_target
        x_tar = target_mat[:, 0] - float(np.dot(target_mat[:, 0], z_target)) * z_target
        if float(np.linalg.norm(x_cur)) > 1e-8 and float(np.linalg.norm(x_tar)) > 1e-8:
            x_cur = normalize_vec(x_cur)
            x_tar = normalize_vec(x_tar)
            desired_principal = float(
                math.atan2(float(np.dot(z_target, np.cross(x_cur, x_tar))), float(np.dot(x_cur, x_tar)))
            )
        else:
            desired_principal = self.wrist_spin_target_rad
        desired_spin = self.wrist_spin_target_rad + math.atan2(
            math.sin(desired_principal - self.wrist_spin_target_rad),
            math.cos(desired_principal - self.wrist_spin_target_rad),
        )
        max_step = math.radians(float(args.wrist_spin_speed_deg_s)) * float(args.dt)
        self.wrist_spin_target_rad += float(np.clip(desired_spin - self.wrist_spin_target_rad, -max_step, max_step))
        data.ctrl[self.spin_act_id] = float(self.wrist_spin_target_rad)

        dq_abs_max = float(np.max(np.abs(dq_arm))) if len(dq_arm) else 0.0
        if dq_abs_max > float(args.max_angvel):
            dq_arm *= float(args.max_angvel) / dq_abs_max

        if self.arm_in_deadband:
            self.q_ctrl_target = np.clip(self.q_ctrl_target, q_arm - SETPOINT_TRACKING_BAND, q_arm + SETPOINT_TRACKING_BAND)
        else:
            self.q_ctrl_target = q_arm + dq_arm * float(args.dt)
        self.q_ctrl_target = np.clip(self.q_ctrl_target, self.joint_lower, self.joint_upper)
        data.ctrl[self.actuator_ids] = self.q_ctrl_target
