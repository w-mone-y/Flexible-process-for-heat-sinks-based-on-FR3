"""Skill library: pose math, skill-step primitives and parameterized skill
pose generators for the three-FR3 flexible line.

All skills are generated from perceived poses + type-level knowledge; there
are no taught waypoints anywhere (flexibility requirement, plan section 5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CTRL_PER_FINGER_M = 0.01568627451 / 100.0
SETPOINT_TRACKING_BAND = 0.05

APPROACH_CLEARANCE = 0.10
LIFT_HEIGHT = 0.12
PLACE_DROP = 0.002
GRASP_FINGER_MARGIN = 0.0015

SCREW_HOVER = 0.08
SCREW_TIP_CLEARANCE = 0.003    # screwdriver tip above the screw marker while spinning, m
SCREW_TURNS = 2.5              # plan 5.3: rotate 2-3 turns about the tool z axis
SCREW_SPIN_SPEED_DEG_S = 300.0

PRESS_HOVER = 0.06
PRESS_DEPTH = 0.003            # press-face travel below the component top face, m
PRESS_HOLD_S = 0.4

INSPECT_HOVER = 0.07
INSPECT_XY_TOL = 0.005
INSPECT_YAW_TOL_DEG = 8.0


def gripper_close_ctrl_for_width(grip_width: float) -> float:
    finger_target = max(grip_width / 2.0 - GRASP_FINGER_MARGIN, 0.0)
    return float(np.clip(finger_target / GRIPPER_CTRL_PER_FINGER_M, 0.0, 255.0))


# ---------------------------------------------------------------------------
# Pose math.
# ---------------------------------------------------------------------------
def normalize_quat(quat: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("quaternion norm is too small")
    q = q / n
    if q[0] < 0.0:
        q = -q
    return q


def normalize_vec(vec: list[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("vector norm is too small")
    return v / n


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


def quat_rot_z(angle_rad: float) -> np.ndarray:
    half = 0.5 * float(angle_rad)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def mat_from_quat(quat: np.ndarray) -> np.ndarray:
    import mujoco

    mat = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    import mujoco

    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).reshape(9))
    return normalize_quat(quat)


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


def yaw_from_mat(mat: np.ndarray) -> float:
    r = np.asarray(mat, dtype=float).reshape(3, 3)
    return float(math.atan2(r[1, 0], r[0, 0]))


def rot_z_mat(angle_rad: float) -> np.ndarray:
    c, s = math.cos(float(angle_rad)), math.sin(float(angle_rad))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def angle_diff(a: float, b: float) -> float:
    return float(math.atan2(math.sin(a - b), math.cos(a - b)))


def yaw_error_with_symmetry(yaw: float, target_yaw: float, symmetry_rad: float) -> float:
    err = abs(angle_diff(yaw, target_yaw))
    if symmetry_rad > 1e-9:
        k = round((yaw - target_yaw) / symmetry_rad)
        for kk in (k - 1, k, k + 1):
            err = min(err, abs(angle_diff(yaw, target_yaw + kk * symmetry_rad)))
    return err


def nearest_symmetric_yaw(yaw: float, target_yaw: float, symmetry_rad: float) -> float:
    """The member of the symmetry family of target_yaw closest to yaw."""
    if symmetry_rad <= 1e-9:
        return target_yaw
    k = round((yaw - target_yaw) / symmetry_rad)
    best = target_yaw + k * symmetry_rad
    for kk in (k - 1, k + 1):
        cand = target_yaw + kk * symmetry_rad
        if abs(angle_diff(yaw, cand)) < abs(angle_diff(yaw, best)):
            best = cand
    return best


def top_down_ee_quat(world_yaw: float) -> np.ndarray:
    return quat_from_rpy(math.pi, 0.0, world_yaw)


# ---------------------------------------------------------------------------
# Command / step primitives.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PoseCommand:
    position: np.ndarray
    quat: np.ndarray


def pose_cmd(position: np.ndarray, quat: np.ndarray) -> PoseCommand:
    return PoseCommand(position=np.asarray(position, dtype=float).copy(), quat=normalize_quat(quat))


def pose_array_to_command(values: list[float] | np.ndarray) -> PoseCommand:
    pose = np.asarray(values, dtype=float)
    if pose.shape == (6,):
        return pose_cmd(pose[:3], quat_from_rpy(float(pose[3]), float(pose[4]), float(pose[5])))
    if pose.shape == (7,):
        return pose_cmd(pose[:3], pose[3:7])
    raise ValueError("pose must be [x y z roll pitch yaw] or [x y z qw qx qy qz]")


@dataclass
class SkillStep:
    label: str
    pose: Optional[PoseCommand] = None
    pose_factory: Optional[Callable[[], PoseCommand]] = None
    action: str = ""            # "" | close_gripper | open_gripper | attach | release | press_seat
                                # | tool_dock | tool_undock | fixture_hold
    action_arg: Any = None
    spin_rad: float = 0.0       # rotate about the tool z axis after arrival
    spin_center: Optional[np.ndarray] = None  # world xy point of the spin axis (tool tip)
    dwell_s: float = 0.1
    pos_tol: float = 0.004
    ori_tol: float = 0.05

    def resolve_pose(self) -> PoseCommand:
        if self.pose is not None:
            return self.pose
        if self.pose_factory is not None:
            return self.pose_factory()
        raise ValueError(f"skill step {self.label} has no pose")


# ---------------------------------------------------------------------------
# Grasp / place / tool-offset pose generators.
# ---------------------------------------------------------------------------
def compute_grasp_ee_pose(object_pos: np.ndarray, object_yaw: float, spec: Any) -> tuple[np.ndarray, np.ndarray]:
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
    """Invert the recorded grasp transform: T_WS = T_WO_target * inv(T_SO)."""
    object_mat = rot_z_mat(object_target_yaw)
    ee_mat = object_mat @ np.asarray(attach_mat_site_obj, dtype=float).T
    ee_pos = np.asarray(object_target_pos, dtype=float) - ee_mat @ np.asarray(attach_pos_site_obj, dtype=float)
    return ee_pos, mat_to_quat(ee_mat)


def compute_tool_ee_pose(
    tip_target_pos: np.ndarray,
    ee_quat: np.ndarray,
    tool_offset_site: np.ndarray,
) -> np.ndarray:
    """Tool-offset inverse: EE (site) position that puts a rigidly-mounted tool
    tip (offset expressed in the EE-site frame) at tip_target_pos with the EE
    orientation ee_quat."""
    r_ee = mat_from_quat(ee_quat)
    return np.asarray(tip_target_pos, dtype=float) - r_ee @ np.asarray(tool_offset_site, dtype=float)


# ---------------------------------------------------------------------------
# Skill step generators. Callers supply perception results; nothing is taught.
# ---------------------------------------------------------------------------
def pick_steps(
    instance: str,
    spec: Any,
    object_pos: np.ndarray,
    object_yaw: float,
) -> tuple[list[SkillStep], dict[str, Any]]:
    """approach -> descend -> close -> attach -> lift, from the perceived pose."""
    grasp_pos, grasp_quat = compute_grasp_ee_pose(object_pos, object_yaw, spec)
    approach_pos = grasp_pos + np.asarray([0.0, 0.0, APPROACH_CLEARANCE])
    lift_pos = grasp_pos + np.asarray([0.0, 0.0, LIFT_HEIGHT])
    record: dict[str, Any] = {}
    steps = [
        SkillStep("approach", pose=pose_cmd(approach_pos, grasp_quat)),
        SkillStep("descend", pose=pose_cmd(grasp_pos, grasp_quat), pos_tol=0.003),
        SkillStep("grasp_close", pose=pose_cmd(grasp_pos, grasp_quat), action="close_gripper", action_arg=spec.grip_width, dwell_s=0.45),
        SkillStep("grasp_attach", pose=pose_cmd(grasp_pos, grasp_quat), action="attach", action_arg=(instance, record), dwell_s=0.15),
        SkillStep("lift", pose=pose_cmd(lift_pos, grasp_quat)),
    ]
    return steps, record


def place_steps(
    instance: str,
    spec: Any,
    target_pose_getter: Callable[[], tuple[np.ndarray, float]],
    attach_record: dict[str, Any],
    loose_tol: bool = False,
    release_dwell_s: Optional[float] = None,
    seat_on_release: bool = False,
) -> list[SkillStep]:
    """transfer -> place -> open fingers -> detach weld -> retreat.

    Order is critical: fingers must open and the grasp weld must be cleared
    BEFORE the arm lifts, otherwise the part rides up and floats in mid-air.

    When seat_on_release is True (Arm2 board placement), a seat action snaps
    the part onto the slot pose right after detach so it cannot drift."""

    def place_ee_pose() -> tuple[np.ndarray, np.ndarray]:
        target_pos, target_yaw = target_pose_getter()
        object_target_pos = target_pos + np.asarray([0.0, 0.0, float(spec.half_height) + float(spec.place_drop)])
        return compute_place_ee_pose(object_target_pos, target_yaw, attach_record["pos_site_obj"], attach_record["mat_site_obj"])

    def place_pose_cmd() -> PoseCommand:
        pos, quat = place_ee_pose()
        return pose_cmd(pos, quat)

    def transfer_pose_cmd() -> PoseCommand:
        pos, quat = place_ee_pose()
        return pose_cmd(pos + np.asarray([0.0, 0.0, APPROACH_CLEARANCE]), quat)

    dwell = float(spec.release_dwell_s) if release_dwell_s is None else float(release_dwell_s)
    # Fingers need time to fully open before the weld is cut and the arm lifts.
    open_dwell = max(dwell, 0.55)
    pos_tol = 0.005 if loose_tol else 0.003
    steps = [
        SkillStep("transfer", pose_factory=transfer_pose_cmd),
        SkillStep("place", pose_factory=place_pose_cmd, pos_tol=pos_tol),
        SkillStep(
            "release_open",
            pose_factory=place_pose_cmd,
            action="open_gripper",
            dwell_s=open_dwell,
            pos_tol=pos_tol,
        ),
        SkillStep(
            "release_detach",
            pose_factory=place_pose_cmd,
            action="release",
            action_arg=instance,
            dwell_s=0.25,
            pos_tol=pos_tol,
        ),
    ]
    if seat_on_release:
        steps.append(
            SkillStep(
                "release_seat",
                pose_factory=place_pose_cmd,
                action="place_seat",
                action_arg=instance,
                dwell_s=0.15,
                pos_tol=pos_tol,
            )
        )
    steps.append(SkillStep("retreat", pose_factory=transfer_pose_cmd, dwell_s=0.2))
    return steps


def press_steps(
    instance: str,
    component_top_pos: np.ndarray,
    slot_yaw: float,
    press_offset_site: np.ndarray,
    snap_arg: Any,
    fixture_after: bool = False,
) -> list[SkillStep]:
    """Press-fit process (plan: independent stage using the press head).

    The press face is aligned to the component top center, pushed down
    PRESS_DEPTH, held, then retracted. On the hold step the positioning-
    fixture snap seats the component.

    When fixture_after is True (Arm2 quick-change flow) the board fixture
    weld is activated after press so the part survives the tool swap."""
    quat = top_down_ee_quat(slot_yaw)
    top = np.asarray(component_top_pos, dtype=float)
    hover_ee = compute_tool_ee_pose(top + np.asarray([0.0, 0.0, PRESS_HOVER]), quat, press_offset_site)
    touch_ee = compute_tool_ee_pose(top + np.asarray([0.0, 0.0, 0.001]), quat, press_offset_site)
    press_ee = compute_tool_ee_pose(top - np.asarray([0.0, 0.0, PRESS_DEPTH]), quat, press_offset_site)
    steps = [
        SkillStep("press_approach", pose=pose_cmd(hover_ee, quat)),
        SkillStep("press_descend", pose=pose_cmd(touch_ee, quat), pos_tol=0.003),
        SkillStep("press_hold", pose=pose_cmd(press_ee, quat), action="press_seat", action_arg=snap_arg, dwell_s=PRESS_HOLD_S, pos_tol=0.006, ori_tol=0.03),
        SkillStep("press_retreat", pose=pose_cmd(hover_ee, quat), dwell_s=0.1),
    ]
    if fixture_after:
        steps.append(
            SkillStep(
                "fixture_hold",
                pose=pose_cmd(hover_ee, quat),
                action="fixture_hold",
                action_arg=instance,
                dwell_s=0.1,
            )
        )
    return steps


def screw_steps(
    screw_pos: np.ndarray,
    screw_yaw: float,
    driver_offset_site: np.ndarray,
) -> list[SkillStep]:
    """Electric-screwdriver fastening: the driver TCP (bit tip) is aligned
    over the screw, lowered onto it, then the wrist spins 2.5 turns. With the
    quick-change screwdriver the bit is coaxial with the tool z axis, so the
    spin turns the bit in place; spin_center still guards against any radial
    TCP offset."""
    quat = top_down_ee_quat(screw_yaw)
    tip_touch = np.asarray(screw_pos, dtype=float) + np.asarray([0.0, 0.0, SCREW_TIP_CLEARANCE])
    hover_ee = compute_tool_ee_pose(tip_touch + np.asarray([0.0, 0.0, SCREW_HOVER]), quat, driver_offset_site)
    touch_ee = compute_tool_ee_pose(tip_touch, quat, driver_offset_site)
    center = np.asarray([float(screw_pos[0]), float(screw_pos[1])], dtype=float)
    return [
        SkillStep("screw_approach", pose=pose_cmd(hover_ee, quat)),
        SkillStep("screw_descend", pose=pose_cmd(touch_ee, quat), pos_tol=0.003),
        SkillStep(
            "screw_spin",
            pose=pose_cmd(touch_ee, quat),
            spin_rad=SCREW_TURNS * 2.0 * math.pi,
            spin_center=center,
            dwell_s=0.2,
            pos_tol=0.003,
        ),
        SkillStep("screw_retreat", pose=pose_cmd(hover_ee, quat), dwell_s=0.1),
    ]


def inspect_steps(slot_pos: np.ndarray, slot_yaw: float) -> list[SkillStep]:
    hover = np.asarray(slot_pos, dtype=float) + np.asarray([0.0, 0.0, INSPECT_HOVER])
    quat = top_down_ee_quat(slot_yaw)
    return [
        SkillStep("inspect_approach", pose=pose_cmd(hover, quat)),
        SkillStep("inspect_hold", pose=pose_cmd(hover, quat), dwell_s=0.8),
        SkillStep("inspect_retreat", pose=pose_cmd(hover + np.asarray([0.0, 0.0, 0.04]), quat), dwell_s=0.1),
    ]
