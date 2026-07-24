"""Robot-independent pose, path and FR3 motion primitives.

The controller intentionally exposes a full SE(3) target while solving the
FR3 task as position + tool-axis (5D) and closing tool roll in the remaining
Jacobian null-space.  This keeps dispensing/camera roll independent without
adding a fictitious joint to the upstream FR3 model.

All distances are metres, angles radians and quaternions scalar-first (wxyz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray
HOME_QPOS = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)


def _unit(vector: ArrayLike, fallback: ArrayLike | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm > 1e-12:
        return value / norm
    if fallback is None:
        raise ValueError("cannot normalize a zero vector")
    return _unit(fallback)


def _solve_small_spd(matrix: ArrayLike, right_hand_side: ArrayLike) -> np.ndarray:
    """Solve one 5x5/6x6 SPD system without high-overhead BLAS dispatch.

    DLS normal matrices are symmetric positive definite because they include
    a positive damping diagonal.  Some macOS NumPy/BLAS builds spend tens of
    milliseconds dispatching ``linalg.solve`` for these tiny matrices.  This
    fixed-size Cholesky solve avoids that thread-dispatch overhead.
    """

    system = np.asarray(matrix, dtype=float)
    rhs = np.asarray(right_hand_side, dtype=float)
    if system.ndim != 2 or system.shape[0] != system.shape[1]:
        raise np.linalg.LinAlgError("SPD system must be square")
    size = int(system.shape[0])
    rhs_was_vector = rhs.ndim == 1
    values = rhs.reshape(size, 1) if rhs_was_vector else rhs.copy()
    if values.ndim != 2 or values.shape[0] != size:
        raise np.linalg.LinAlgError("right-hand side has incompatible dimensions")

    lower = np.zeros_like(system)
    for row in range(size):
        for column in range(row + 1):
            residual = float(system[row, column] - np.dot(lower[row, :column], lower[column, :column]))
            if row == column:
                if residual <= 1.0e-15 or not np.isfinite(residual):
                    raise np.linalg.LinAlgError("SPD Cholesky factorisation failed")
                lower[row, column] = math.sqrt(residual)
            else:
                lower[row, column] = residual / lower[column, column]

    forward = np.zeros_like(values)
    for row in range(size):
        forward[row] = (values[row] - lower[row, :row] @ forward[:row]) / lower[row, row]
    solution = np.zeros_like(values)
    for row in range(size - 1, -1, -1):
        solution[row] = (forward[row] - lower[row + 1 :, row] @ solution[row + 1 :]) / lower[row, row]
    return solution[:, 0] if rhs_was_vector else solution


def normalize_quat(quaternion: ArrayLike) -> np.ndarray:
    quat = _unit(quaternion)
    if quat.shape != (4,):
        raise ValueError("quaternion must contain four values in wxyz order")
    # q and -q encode the same rotation. Canonicalization makes comparisons and
    # serialized snapshots deterministic.
    return -quat if quat[0] < 0.0 else quat


def quat_conjugate(quaternion: ArrayLike) -> np.ndarray:
    w, x, y, z = normalize_quat(quaternion)
    return np.asarray([w, -x, -y, -z], dtype=float)


def quat_multiply(left: ArrayLike, right: ArrayLike) -> np.ndarray:
    aw, ax, ay, az = np.asarray(left, dtype=float)
    bw, bx, by, bz = np.asarray(right, dtype=float)
    return normalize_quat(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_to_matrix(quaternion: ArrayLike) -> np.ndarray:
    w, x, y, z = normalize_quat(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat(matrix: ArrayLike) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a scalar-first quaternion."""

    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * s,
            (matrix[2, 1] - matrix[1, 2]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        ]
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(max(1e-16, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quat = [
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            ]
        elif index == 1:
            s = math.sqrt(max(1e-16, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            quat = [
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            ]
        else:
            s = math.sqrt(max(1e-16, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            quat = [
                (matrix[1, 0] - matrix[0, 1]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
            ]
    return normalize_quat(quat)


def quat_slerp(left: ArrayLike, right: ArrayLike, fraction: float) -> np.ndarray:
    q0 = normalize_quat(left)
    q1 = normalize_quat(right)
    amount = float(np.clip(fraction, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quat(q0 + amount * (q1 - q0))
    angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
    scale = math.sin(angle)
    return normalize_quat(
        math.sin((1.0 - amount) * angle) / scale * q0 + math.sin(amount * angle) / scale * q1
    )


def signed_twist_error(current: np.ndarray, target: np.ndarray, axis: np.ndarray) -> float:
    """Signed rotation from current X to target X around ``axis``."""

    axis = _unit(axis, [0.0, 0.0, 1.0])
    current_x = current[:, 0] - float(np.dot(current[:, 0], axis)) * axis
    target_x = target[:, 0] - float(np.dot(target[:, 0], axis)) * axis
    if np.linalg.norm(current_x) < 1e-9 or np.linalg.norm(target_x) < 1e-9:
        return 0.0
    current_x = _unit(current_x)
    target_x = _unit(target_x)
    return math.atan2(float(np.dot(axis, np.cross(current_x, target_x))), float(np.dot(current_x, target_x)))


@dataclass(frozen=True)
class Pose:
    """Immutable SE(3) pose with scalar-first quaternion."""

    position: np.ndarray
    quaternion: np.ndarray = field(default_factory=lambda: np.asarray([1.0, 0.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite xyz vector")
        quaternion = normalize_quat(self.quaternion)
        if not np.all(np.isfinite(quaternion)):
            raise ValueError("quaternion must be finite")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "quaternion", quaternion.copy())

    @property
    def quat(self) -> np.ndarray:
        """Compatibility alias used by the older controller."""

        return self.quaternion

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.quaternion)

    @property
    def matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.position
        return transform

    def transformed(self, local: "Pose") -> "Pose":
        rotation = self.rotation
        return Pose(
            self.position + rotation @ local.position, quat_multiply(self.quaternion, local.quaternion)
        )

    def inverse(self) -> "Pose":
        inverse_quat = quat_conjugate(self.quaternion)
        inverse_rotation = quat_to_matrix(inverse_quat)
        return Pose(-(inverse_rotation @ self.position), inverse_quat)

    def interpolate(self, other: "Pose", fraction: float) -> "Pose":
        amount = float(np.clip(fraction, 0.0, 1.0))
        return Pose(
            self.position + amount * (other.position - self.position),
            quat_slerp(self.quaternion, other.quaternion, amount),
        )

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def from_matrix(cls, transform: ArrayLike) -> "Pose":
        matrix = np.asarray(transform, dtype=float).reshape(4, 4)
        return cls(matrix[:3, 3], matrix_to_quat(matrix[:3, :3]))


def pose_from_site(data: Any, site: int | str) -> Pose:
    view = data.site(site)
    return Pose(np.asarray(view.xpos, dtype=float), matrix_to_quat(np.asarray(view.xmat).reshape(3, 3)))


@dataclass(frozen=True)
class PolylineTrajectory:
    """Piecewise-linear, constant translational-speed SE(3) trajectory."""

    waypoints: tuple[Pose, ...]
    speed_m_s: float = 0.04
    sample_spacing_m: float = 0.01

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("a trajectory requires at least two waypoints")
        if self.speed_m_s <= 0.0:
            raise ValueError("speed_m_s must be positive")
        if not 0.0 < self.sample_spacing_m <= 0.01 + 1e-12:
            raise ValueError("sample spacing must be in (0, 0.01] m")

    @property
    def segment_lengths(self) -> np.ndarray:
        return np.asarray(
            [
                np.linalg.norm(right.position - left.position)
                for left, right in zip(self.waypoints, self.waypoints[1:])
            ],
            dtype=float,
        )

    @property
    def length_m(self) -> float:
        return float(np.sum(self.segment_lengths))

    @property
    def duration_s(self) -> float:
        return self.length_m / self.speed_m_s

    def sample_distance(self, distance_m: float) -> Pose:
        lengths = self.segment_lengths
        total = float(np.sum(lengths))
        if total <= 1e-12:
            return self.waypoints[-1]
        distance = float(np.clip(distance_m, 0.0, total))
        consumed = 0.0
        for index, length in enumerate(lengths):
            if distance <= consumed + length or index == len(lengths) - 1:
                fraction = 1.0 if length <= 1e-12 else (distance - consumed) / float(length)
                return self.waypoints[index].interpolate(self.waypoints[index + 1], fraction)
            consumed += float(length)
        return self.waypoints[-1]

    def sample(self, elapsed_s: float) -> Pose:
        return self.sample_distance(max(0.0, float(elapsed_s)) * self.speed_m_s)

    def samples(self, spacing_m: float | None = None) -> tuple[Pose, ...]:
        spacing = self.sample_spacing_m if spacing_m is None else float(spacing_m)
        if not 0.0 < spacing <= 0.01 + 1e-12:
            raise ValueError("reachability sampling interval must not exceed 10 mm")
        samples = [self.waypoints[0]]
        for left, right, length in zip(self.waypoints, self.waypoints[1:], self.segment_lengths):
            count = max(1, int(math.ceil(float(length) / spacing)))
            samples.extend(left.interpolate(right, index / count) for index in range(1, count + 1))
        return tuple(samples)

    def project(self, position: ArrayLike) -> tuple[float, float]:
        """Return nearest path progress [0,1] and lateral error in metres."""

        point = np.asarray(position, dtype=float)
        lengths = self.segment_lengths
        total = max(float(np.sum(lengths)), 1e-12)
        best_error = math.inf
        best_distance = 0.0
        consumed = 0.0
        for left, right, length in zip(self.waypoints, self.waypoints[1:], lengths):
            delta = right.position - left.position
            if length <= 1e-12:
                fraction = 0.0
                nearest = left.position
            else:
                fraction = float(np.clip(np.dot(point - left.position, delta) / (length * length), 0.0, 1.0))
                nearest = left.position + fraction * delta
            error = float(np.linalg.norm(point - nearest))
            if error < best_error:
                best_error = error
                best_distance = consumed + fraction * float(length)
            consumed += float(length)
        return best_distance / total, best_error


def safe_corridor_trajectory(
    start: Pose,
    goal: Pose,
    *,
    clearance_z: float = 0.50,
    speed_m_s: float = 0.04,
    sample_spacing_m: float = 0.01,
) -> PolylineTrajectory:
    """Generate approach-lift-safe-corridor-descend motion."""

    safe_z = max(float(clearance_z), float(start.position[2]), float(goal.position[2]))
    start_lift = Pose([start.position[0], start.position[1], safe_z], start.quaternion)
    goal_lift = Pose([goal.position[0], goal.position[1], safe_z], goal.quaternion)
    waypoints = [start]
    if np.linalg.norm(start_lift.position - start.position) > 1e-9:
        waypoints.append(start_lift)
    if np.linalg.norm(goal_lift.position - waypoints[-1].position) > 1e-9:
        waypoints.append(goal_lift)
    if np.linalg.norm(goal.position - waypoints[-1].position) > 1e-9 or not np.allclose(
        goal.quaternion, waypoints[-1].quaternion
    ):
        waypoints.append(goal)
    if len(waypoints) == 1:
        waypoints.append(goal)
    return PolylineTrajectory(tuple(waypoints), speed_m_s, sample_spacing_m)


@dataclass(frozen=True)
class MotionConfig:
    dt: float = 0.002
    damping: float = 0.03
    position_gain: float = 30.0
    axis_gain: float = 8.0
    roll_gain: float = 6.0
    joint_center_gain: float = 0.03
    max_joint_speed: float = 1.8
    max_command_lead_rad: float = 0.08
    position_servo_lead_s: float = 0.12
    position_tolerance_m: float = 0.003
    orientation_tolerance_rad: float = math.radians(3.0)
    path_rmse_limit_m: float = 0.003
    path_max_error_limit_m: float = 0.005


@dataclass(frozen=True)
class ReachabilityResult:
    reachable: bool
    position_error_m: float
    orientation_error_rad: float
    joint_positions: np.ndarray
    iterations: int
    reason: str = ""


class ArmController:
    """One prefixed FR3 controller using 5D DLS plus independent tool roll."""

    def __init__(self, model: Any, data: Any, prefix: str, config: MotionConfig | None = None) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.prefix = prefix.rstrip("_")
        self.config = config or MotionConfig(dt=float(model.opt.timestep))
        stem = self.prefix + "_"
        self.site_id = int(model.site(stem + "attachment_site").id)
        target_body_id = int(model.body(stem + "target").id)
        self.mocap_id = int(model.body_mocapid[target_body_id])
        if self.mocap_id < 0:
            raise RuntimeError(f"{stem}target is not a mocap body")
        joint_ids = np.asarray([int(model.joint(f"{stem}fr3_joint{i}").id) for i in range(1, 8)], dtype=int)
        self.joint_ids = joint_ids
        self.qpos_ids = np.asarray([int(model.jnt_qposadr[index]) for index in joint_ids], dtype=int)
        self.dof_ids = np.asarray([int(model.jnt_dofadr[index]) for index in joint_ids], dtype=int)
        self.actuator_ids = np.asarray(
            [int(model.actuator(f"{stem}fr3_joint{i}").id) for i in range(1, 8)], dtype=int
        )
        self.lower = np.asarray([model.jnt_range[index, 0] for index in joint_ids], dtype=float)
        self.upper = np.asarray([model.jnt_range[index, 1] for index in joint_ids], dtype=float)
        self.mid = 0.5 * (self.lower + self.upper)
        self.half_range = np.maximum(0.5 * (self.upper - self.lower), 1e-6)
        self.jacobian = np.zeros((6, model.nv), dtype=float)
        self.identity = np.eye(7)
        self.q_command = np.asarray(data.qpos[self.qpos_ids], dtype=float).copy()
        self.target = pose_from_site(data, self.site_id)
        self.tcp_target: Pose | None = None
        self.tool_transform = Pose.identity()  # flange -> active TCP
        self.last_position_error_m = 0.0
        self.last_orientation_error_rad = 0.0
        self.last_roll_error_rad = 0.0
        self.locked_local_indices: tuple[int, ...] = ()
        self.full_orientation = False
        self.enabled = True
        self.failure = ""
        self.set_target(self.target)

    @property
    def name(self) -> str:
        return self.prefix

    def reset(self, home: ArrayLike = HOME_QPOS) -> None:
        values = np.asarray(home, dtype=float)
        if values.shape != (7,):
            raise ValueError("FR3 home configuration must contain seven joint positions")
        values = np.clip(values, self.lower, self.upper)
        self.data.qpos[self.qpos_ids] = values
        self.data.qvel[self.dof_ids] = 0.0
        self.data.ctrl[self.actuator_ids] = values
        self.q_command = values.copy()
        self.mujoco.mj_forward(self.model, self.data)
        self.target = self.current_flange_pose()
        self.set_target(self.target)
        self.enabled = True
        self.locked_local_indices = ()
        self.full_orientation = False
        self.failure = ""

    def current_flange_pose(self) -> Pose:
        return pose_from_site(self.data, self.site_id)

    def current_tcp_pose(self) -> Pose:
        return self.current_flange_pose().transformed(self.tool_transform)

    def set_tool_transform(self, transform: Pose | None) -> None:
        self.tool_transform = Pose.identity() if transform is None else transform

    def set_target(self, pose: Pose, *, tcp: bool = False) -> None:
        # The DLS site is the FR3 attachment site. Convert a requested TCP pose
        # back to the corresponding flange pose when a tool is active.
        self.tcp_target = pose if tcp else None
        self.target = pose.transformed(self.tool_transform.inverse()) if tcp else pose
        self.data.mocap_pos[self.mocap_id] = self.target.position
        self.data.mocap_quat[self.mocap_id] = self.target.quaternion

    def hold(self) -> None:
        self.set_target(self.current_flange_pose())
        self.q_command = np.asarray(self.data.qpos[self.qpos_ids], dtype=float).copy()

    def stop(self, reason: str = "stopped") -> None:
        self.enabled = False
        self.failure = str(reason)
        self.hold()

    def _task_velocity(
        self, pose: Pose, *, tcp: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        current = self.current_tcp_pose() if tcp else self.current_flange_pose()
        current_rotation = current.rotation
        target_rotation = pose.rotation
        position_error = pose.position - current.position
        current_axis = _unit(current_rotation[:, 2])
        target_axis = _unit(target_rotation[:, 2])
        cross = np.cross(current_axis, target_axis)
        cross_norm = float(np.linalg.norm(cross))
        axis_angle = math.atan2(cross_norm, float(np.clip(np.dot(current_axis, target_axis), -1.0, 1.0)))
        axis_vector = np.zeros(3) if cross_norm < 1e-10 else cross / cross_norm * axis_angle
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, current_axis))) > 0.85:
            reference = np.asarray([0.0, 1.0, 0.0])
        basis1 = _unit(reference - np.dot(reference, current_axis) * current_axis)
        basis2 = _unit(np.cross(current_axis, basis1))
        task = np.zeros(5)
        task[:3] = self.config.position_gain * position_error
        omega = self.config.axis_gain * axis_vector
        task[3:] = [np.dot(basis1, omega), np.dot(basis2, omega)]
        roll_error = signed_twist_error(current_rotation, target_rotation, target_axis)
        return task, basis1, basis2, float(np.linalg.norm(position_error)), axis_angle, roll_error

    def _joint_velocity(
        self,
        pose: Pose,
        *,
        tcp: bool = False,
        locked_local_indices: Iterable[int] = (),
        full_orientation: bool = False,
    ) -> tuple[np.ndarray, float, float, float]:
        """Return one DLS step while optionally keeping selected joints fixed.

        A locked joint must be removed from the Jacobian solve itself.  Solving
        with all seven joints and overwriting one column afterward invalidates
        the full-orientation solution and lets a mounted tool rotate underneath
        a rigidly carried workpiece.
        """

        locked = frozenset(int(index) for index in locked_local_indices)
        if any(index < 0 or index >= 7 for index in locked):
            raise ValueError("locked joint indices must be in the range 0..6")
        active = np.asarray([index for index in range(7) if index not in locked], dtype=int)
        if active.size < 6:
            raise ValueError("full SE(3) IK requires at least six active joints")
        self.mujoco.mj_jacSite(self.model, self.data, self.jacobian[:3], self.jacobian[3:], self.site_id)
        task, basis1, basis2, position_error, axis_error, roll_error = self._task_velocity(pose, tcp=tcp)
        jac_pos_full = self.jacobian[:3, self.dof_ids]
        jac_rot_full = self.jacobian[3:, self.dof_ids]
        if tcp:
            flange = self.current_flange_pose()
            offset = flange.rotation @ self.tool_transform.position
            skew = np.asarray(
                [
                    [0.0, -offset[2], offset[1]],
                    [offset[2], 0.0, -offset[0]],
                    [-offset[1], offset[0], 0.0],
                ]
            )
            jac_pos_full = jac_pos_full - skew @ jac_rot_full
        jac_pos = jac_pos_full[:, active]
        jac_rot = jac_rot_full[:, active]
        if locked or full_orientation:
            # With one joint mechanically fixed there are exactly six active
            # coordinates.  Solve the complete spatial task directly; a 5D
            # solve followed by a damped null-space roll correction leaves a
            # small orientation residual that accumulates along a carry path.
            current = self.current_tcp_pose() if tcp else self.current_flange_pose()
            current_axis = _unit(current.rotation[:, 2])
            target_axis = _unit(pose.rotation[:, 2])
            axis_cross = np.cross(current_axis, target_axis)
            cross_norm = float(np.linalg.norm(axis_cross))
            axis_angle = math.atan2(
                cross_norm,
                float(np.clip(np.dot(current_axis, target_axis), -1.0, 1.0)),
            )
            axis_vector = np.zeros(3) if cross_norm < 1e-10 else axis_cross / cross_norm * axis_angle
            angular_task = (
                self.config.axis_gain * axis_vector + self.config.roll_gain * roll_error * target_axis
            )
            task6 = np.concatenate((task[:3], angular_task))
            jac6 = np.vstack((jac_pos, jac_rot))
            regularizer6 = (self.config.damping**2) * np.eye(6)
            try:
                inverse6 = jac6.T @ _solve_small_spd(
                    jac6 @ jac6.T + regularizer6,
                    np.eye(6),
                )
            except np.linalg.LinAlgError:
                return np.zeros(7), position_error, axis_error, roll_error
            active_velocity = inverse6 @ task6
            if active.size > 6:
                nullspace6 = np.eye(active.size) - inverse6 @ jac6
                qpos = np.asarray(self.data.qpos[self.qpos_ids], dtype=float)
                centered = self.config.joint_center_gain * (self.mid - qpos) / self.half_range
                active_velocity += nullspace6 @ centered[active]
            velocity = np.zeros(7, dtype=float)
            velocity[active] = active_velocity
            peak = float(np.max(np.abs(velocity)))
            if peak > self.config.max_joint_speed:
                velocity *= self.config.max_joint_speed / peak
            return velocity, position_error, axis_error, roll_error

        jac5 = np.vstack((jac_pos, np.vstack((basis1, basis2)) @ jac_rot))
        regularizer = (self.config.damping**2) * np.eye(5)
        try:
            inverse = jac5.T @ _solve_small_spd(jac5 @ jac5.T + regularizer, np.eye(5))
        except np.linalg.LinAlgError:
            return np.zeros(7), position_error, axis_error, roll_error
        active_velocity = inverse @ task
        nullspace = np.eye(active.size) - inverse @ jac5

        # Full-orientation twist is solved after the 5D task, through its
        # remaining null-space. This is the independent tool-roll channel.
        target_axis = pose.rotation[:, 2]
        projected_roll_jac = (target_axis @ jac_rot @ nullspace).reshape(1, active.size)
        denominator = (projected_roll_jac @ projected_roll_jac.T).item() + self.config.damping**2
        if denominator > 1e-10:
            active_velocity += (nullspace @ projected_roll_jac.T).ravel() * (
                self.config.roll_gain * roll_error / denominator
            )

        qpos = np.asarray(self.data.qpos[self.qpos_ids], dtype=float)
        centered = self.config.joint_center_gain * (self.mid - qpos) / self.half_range
        active_velocity += nullspace @ centered[active]
        velocity = np.zeros(7, dtype=float)
        velocity[active] = active_velocity
        peak = float(np.max(np.abs(velocity)))
        if peak > self.config.max_joint_speed:
            velocity *= self.config.max_joint_speed / peak
        return velocity, position_error, axis_error, roll_error

    def control_tick(self, dt: float | None = None) -> None:
        if not self.enabled:
            self.data.ctrl[self.actuator_ids] = self.q_command
            return
        timestep = self.config.dt if dt is None else float(dt)
        target = self.tcp_target if self.tcp_target is not None else self.target
        velocity, position_error, orientation_error, roll_error = self._joint_velocity(
            target,
            tcp=self.tcp_target is not None,
            locked_local_indices=self.locked_local_indices,
            full_orientation=self.full_orientation,
        )
        qpos = np.asarray(self.data.qpos[self.qpos_ids], dtype=float)
        # Convert the IK velocity to a position-servo target with a short
        # prediction horizon. A one-timestep lead is too small for the FR3
        # actuator damping (kp/kv ~= 10/s) and causes centimetres of path lag;
        # 100 ms makes the closed-loop joint velocity closely follow dq.
        lead_time = max(timestep, float(self.config.position_servo_lead_s))
        command = qpos + velocity * lead_time
        lead = float(self.config.max_command_lead_rad)
        command = np.clip(command, qpos - lead, qpos + lead)
        self.q_command = np.clip(command, self.lower, self.upper)
        self.data.ctrl[self.actuator_ids] = self.q_command
        self.last_position_error_m = position_error
        self.last_orientation_error_rad = orientation_error
        self.last_roll_error_rad = roll_error

    @property
    def at_target(self) -> bool:
        return (
            self.last_position_error_m <= self.config.position_tolerance_m
            and self.last_orientation_error_rad <= self.config.orientation_tolerance_rad
            and abs(self.last_roll_error_rad) <= self.config.orientation_tolerance_rad
        )

    def _save_kinematic_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Pose, Pose | None]:
        return (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.ctrl.copy(),
            self.target,
            self.tcp_target,
        )

    def _restore_kinematic_state(
        self,
        state: tuple[np.ndarray, np.ndarray, np.ndarray, Pose, Pose | None],
    ) -> None:
        saved_qpos, saved_qvel, saved_ctrl, saved_target, saved_tcp_target = state
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.data.ctrl[:] = saved_ctrl
        self.target = saved_target
        self.tcp_target = saved_tcp_target
        self.mujoco.mj_forward(self.model, self.data)

    def _solve_ik_inplace(
        self,
        target: Pose,
        *,
        tcp: bool = False,
        seed: ArrayLike | None = None,
        max_iterations: int = 600,
        step_s: float = 0.03,
        locked_joints: Mapping[int, float] | None = None,
        position_tolerance_m: float | None = None,
        orientation_tolerance_rad: float | None = None,
        full_orientation: bool = False,
    ) -> ReachabilityResult:
        locks = {int(index): float(value) for index, value in (locked_joints or {}).items()}
        if any(index < 0 or index >= 7 for index in locks):
            raise ValueError("locked joint indices must be in the range 0..6")
        if len(locks) > 1:
            raise ValueError("full SE(3) IK supports at most one locked FR3 joint")
        position_tolerance = (
            self.config.position_tolerance_m if position_tolerance_m is None else float(position_tolerance_m)
        )
        orientation_tolerance = (
            self.config.orientation_tolerance_rad
            if orientation_tolerance_rad is None
            else float(orientation_tolerance_rad)
        )
        if position_tolerance <= 0.0 or orientation_tolerance <= 0.0:
            raise ValueError("IK tolerances must be positive")
        if seed is not None:
            candidate = np.asarray(seed, dtype=float)
            if candidate.shape != (7,):
                raise ValueError("IK seed must have seven joints")
            self.data.qpos[self.qpos_ids] = np.clip(candidate, self.lower, self.upper)
        for index, value in locks.items():
            self.data.qpos[self.qpos_ids[index]] = np.clip(
                value,
                self.lower[index],
                self.upper[index],
            )
        self.mujoco.mj_forward(self.model, self.data)
        position_error = math.inf
        orientation_error = math.inf
        for iteration in range(1, int(max_iterations) + 1):
            velocity, position_error, axis_error, roll_error = self._joint_velocity(
                target,
                tcp=tcp,
                locked_local_indices=locks,
                full_orientation=full_orientation,
            )
            orientation_error = math.hypot(axis_error, roll_error)
            if (
                position_error <= position_tolerance
                and axis_error <= orientation_tolerance
                and abs(roll_error) <= orientation_tolerance
            ):
                return ReachabilityResult(
                    True,
                    position_error,
                    orientation_error,
                    np.asarray(self.data.qpos[self.qpos_ids], dtype=float).copy(),
                    iteration,
                )
            values = np.asarray(self.data.qpos[self.qpos_ids], dtype=float) + velocity * float(step_s)
            for index, value in locks.items():
                values[index] = value
            self.data.qpos[self.qpos_ids] = np.clip(values, self.lower, self.upper)
            self.mujoco.mj_forward(self.model, self.data)
        return ReachabilityResult(
            False,
            position_error,
            orientation_error,
            np.asarray(self.data.qpos[self.qpos_ids], dtype=float).copy(),
            int(max_iterations),
            (
                f"target residual exceeds {position_tolerance * 1000:.3g} mm / "
                f"{math.degrees(orientation_tolerance):.3g} degrees"
            ),
        )

    def solve_ik(
        self,
        target: Pose,
        *,
        tcp: bool = False,
        seed: ArrayLike | None = None,
        max_iterations: int = 600,
        step_s: float = 0.03,
        locked_joints: Mapping[int, float] | None = None,
        position_tolerance_m: float | None = None,
        orientation_tolerance_rad: float | None = None,
        full_orientation: bool = False,
    ) -> ReachabilityResult:
        """Solve one sampled target without modifying the live simulation."""

        saved_state = self._save_kinematic_state()
        try:
            return self._solve_ik_inplace(
                target,
                tcp=tcp,
                seed=seed,
                max_iterations=max_iterations,
                step_s=step_s,
                locked_joints=locked_joints,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                full_orientation=full_orientation,
            )
        finally:
            self._restore_kinematic_state(saved_state)

    def validate_trajectory(
        self,
        trajectory: PolylineTrajectory,
        *,
        tcp: bool = True,
        locked_joints: Mapping[int, float] | None = None,
        position_tolerance_m: float | None = None,
        orientation_tolerance_rad: float | None = None,
        full_orientation: bool = False,
    ) -> tuple[ReachabilityResult, ...]:
        # A trajectory can contain dozens of samples.  Saving, restoring and
        # forwarding the complete MuJoCo state for every sample made path
        # validation needlessly expensive.  Solve the samples in-place so the
        # previous solution is also the natural warm start, then restore the
        # live simulation exactly once at the end.
        saved_state = self._save_kinematic_state()
        results: list[ReachabilityResult] = []
        seed = np.asarray(self.data.qpos[self.qpos_ids], dtype=float).copy()
        try:
            for target in trajectory.samples(min(trajectory.sample_spacing_m, 0.01)):
                result = self._solve_ik_inplace(
                    target,
                    tcp=tcp,
                    seed=seed,
                    locked_joints=locked_joints,
                    position_tolerance_m=position_tolerance_m,
                    orientation_tolerance_rad=orientation_tolerance_rad,
                    full_orientation=full_orientation,
                )
                results.append(result)
                if not result.reachable:
                    break
                seed = result.joint_positions
            return tuple(results)
        finally:
            self._restore_kinematic_state(saved_state)


class ExecutionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class MotionEvent:
    kind: str
    message: str
    time_s: float


class TrajectoryExecutor:
    """Lag-bounded executor with actual TCP progress and path-error metrics.

    The nominal feed rate is treated as an upper bound.  Commanded progress is
    paused whenever the actual TCP falls more than a small look-ahead distance
    behind the current set point.  This prevents a time-driven set point from
    cutting across fins while the physical arm is still catching up.
    """

    def __init__(self, controller: ArmController) -> None:
        self.controller = controller
        self.state = ExecutionState.IDLE
        self.trajectory: PolylineTrajectory | None = None
        self.started_at = 0.0
        self.deadline = 0.0
        self.last_tick_at = 0.0
        self.command_distance_m = 0.0
        self.actual_progress = 0.0
        self.errors_m: list[float] = []
        self.events: list[MotionEvent] = []
        self.error = ""
        self.last_command_progress_at = 0.0
        self.stall_timeout_s = 10.0
        self.lookahead_m = min(
            self.controller.config.position_tolerance_m,
            0.8 * self.controller.config.path_max_error_limit_m,
        )

    @property
    def rmse_m(self) -> float:
        return math.sqrt(float(np.mean(np.square(self.errors_m)))) if self.errors_m else 0.0

    @property
    def max_error_m(self) -> float:
        return max(self.errors_m, default=0.0)

    def start(self, trajectory: PolylineTrajectory, now_s: float, timeout_s: float | None = None) -> None:
        if self.state is ExecutionState.RUNNING:
            raise RuntimeError("trajectory executor is already running")
        checks = self.controller.validate_trajectory(trajectory)
        failed = next(((index, item) for index, item in enumerate(checks) if not item.reachable), None)
        if failed is not None:
            index, result = failed
            reason = result.reason or "trajectory is unreachable"
            self.fail(
                now_s,
                f"trajectory sample {index + 1}/{len(checks)}: {reason}; "
                f"residual {result.position_error_m * 1000:.2f} mm / "
                f"{math.degrees(result.orientation_error_rad):.2f} deg",
            )
            return
        self.trajectory = trajectory
        self.started_at = float(now_s)
        allowance = max(1.0, 0.5 * trajectory.duration_s)
        duration = trajectory.duration_s + allowance if timeout_s is None else float(timeout_s)
        self.deadline = self.started_at + duration
        self.last_tick_at = self.started_at
        self.last_command_progress_at = self.started_at
        self.command_distance_m = 0.0
        self.actual_progress = 0.0
        self.errors_m.clear()
        self.error = ""
        self.state = ExecutionState.RUNNING
        self.controller.set_target(trajectory.waypoints[0], tcp=True)
        self.events.append(MotionEvent("started", "trajectory started", float(now_s)))

    def tick(self, now_s: float) -> ExecutionState:
        if self.state is not ExecutionState.RUNNING or self.trajectory is None:
            return self.state
        now = float(now_s)
        if now > self.deadline:
            self.fail(now, "trajectory timeout")
            return self.state
        actual = self.controller.current_tcp_pose()
        current_command = self.trajectory.sample_distance(self.command_distance_m)
        tracking_error = float(np.linalg.norm(actual.position - current_command.position))
        delta = max(0.0, now - self.last_tick_at)
        self.last_tick_at = now
        previous_command_distance = self.command_distance_m
        if tracking_error <= self.lookahead_m:
            self.command_distance_m = min(
                self.trajectory.length_m,
                self.command_distance_m + self.trajectory.speed_m_s * delta,
            )
        if self.command_distance_m > previous_command_distance + 1e-9:
            self.last_command_progress_at = now
        elif (
            self.command_distance_m < self.trajectory.length_m - 1e-9
            and now - self.last_command_progress_at > self.stall_timeout_s
        ):
            self.fail(
                now,
                f"trajectory tracking stalled for {self.stall_timeout_s:.1f} s "
                f"at {tracking_error * 1000:.2f} mm error",
            )
            return self.state
        desired = self.trajectory.sample_distance(self.command_distance_m)
        self.controller.set_target(desired, tcp=True)
        progress, _lateral_error = self.trajectory.project(actual.position)
        self.actual_progress = max(self.actual_progress, progress)
        # KPI error is time-synchronized TCP tracking error, not merely the
        # distance to any point on the geometric polyline (which can conceal a
        # robot lagging far behind the commanded progress).
        self.errors_m.append(float(np.linalg.norm(actual.position - desired.position)))
        command_complete = self.command_distance_m >= self.trajectory.length_m - 1e-9
        if command_complete and self.controller.at_target:
            if self.rmse_m > self.controller.config.path_rmse_limit_m:
                self.fail(now, f"path RMSE {self.rmse_m * 1000:.2f} mm exceeds 3 mm")
            elif self.max_error_m > self.controller.config.path_max_error_limit_m:
                self.fail(now, f"path maximum error {self.max_error_m * 1000:.2f} mm exceeds 5 mm")
            else:
                self.state = ExecutionState.COMPLETE
                self.actual_progress = 1.0
                self.events.append(MotionEvent("completed", "trajectory completed", now))
        return self.state

    def fail(self, now_s: float, reason: str) -> None:
        # A failure is emitted exactly once even if the outer actor continues
        # ticking after it has entered ERROR.
        if self.state is ExecutionState.ERROR:
            return
        self.state = ExecutionState.ERROR
        self.error = str(reason)
        self.controller.stop(self.error)
        self.events.append(MotionEvent("failed", self.error, float(now_s)))

    def cancel(self, now_s: float = 0.0) -> None:
        self.state = ExecutionState.CANCELLED
        self.controller.hold()
        self.events.append(MotionEvent("cancelled", "trajectory cancelled", float(now_s)))


@dataclass(frozen=True)
class SkillStep:
    label: str
    trajectory: PolylineTrajectory | None = None
    action: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    dwell_s: float = 0.0
    timeout_s: float | None = None


class SkillExecutor:
    """Small strict step machine; unknown actions fail immediately."""

    def __init__(
        self,
        trajectory_executor: TrajectoryExecutor,
        actions: Mapping[str, Callable[[Mapping[str, Any]], None]] | None = None,
    ) -> None:
        self.motion = trajectory_executor
        self.actions = dict(actions or {})
        self.steps: tuple[SkillStep, ...] = ()
        self.index = 0
        self.state = ExecutionState.IDLE
        self.next_time = 0.0
        self.error = ""

    def start(self, steps: Iterable[SkillStep], now_s: float) -> None:
        self.steps = tuple(steps)
        if not self.steps:
            raise ValueError("skill contains no steps")
        self.index = 0
        self.state = ExecutionState.RUNNING
        self.error = ""
        self._enter(float(now_s))

    def _enter(self, now: float) -> None:
        step = self.steps[self.index]
        if step.action:
            callback = self.actions.get(step.action)
            if callback is None:
                self.error = f"unknown action: {step.action}"
                self.state = ExecutionState.ERROR
                self.motion.fail(now, self.error)
                return
            try:
                callback(step.payload)
            except Exception as exc:  # action boundary intentionally normalizes actor failures
                self.error = f"action {step.action} failed: {exc}"
                self.state = ExecutionState.ERROR
                self.motion.fail(now, self.error)
                return
        if step.trajectory is not None:
            self.motion.start(step.trajectory, now, step.timeout_s)
            if self.motion.state is ExecutionState.ERROR:
                self.error = self.motion.error
                self.state = ExecutionState.ERROR
        else:
            self.next_time = now + max(0.0, float(step.dwell_s))

    def tick(self, now_s: float) -> ExecutionState:
        if self.state is not ExecutionState.RUNNING:
            return self.state
        step = self.steps[self.index]
        finished = False
        if step.trajectory is not None:
            motion_state = self.motion.tick(now_s)
            if motion_state is ExecutionState.ERROR:
                self.error = self.motion.error
                self.state = ExecutionState.ERROR
                return self.state
            finished = motion_state is ExecutionState.COMPLETE
        else:
            finished = float(now_s) >= self.next_time
        if not finished:
            return self.state
        self.index += 1
        if self.index >= len(self.steps):
            self.state = ExecutionState.COMPLETE
            return self.state
        self._enter(float(now_s))
        return self.state


def default_scene_path() -> Path:
    return Path(__file__).resolve().parent.parent / "brazing_line.xml"


__all__ = [
    "ArmController",
    "ExecutionState",
    "HOME_QPOS",
    "MotionConfig",
    "MotionEvent",
    "PolylineTrajectory",
    "Pose",
    "ReachabilityResult",
    "SkillExecutor",
    "SkillStep",
    "TrajectoryExecutor",
    "default_scene_path",
    "matrix_to_quat",
    "normalize_quat",
    "pose_from_site",
    "quat_conjugate",
    "quat_multiply",
    "quat_slerp",
    "quat_to_matrix",
    "safe_corridor_trajectory",
]
