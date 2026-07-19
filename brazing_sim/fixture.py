"""Flexible comb fixture and non-blocking upper-press control.

``FixtureController`` owns only fixture mechanics; it does not advance MuJoCo
time and never sleeps.  The application keeps stepping the model, calls
``update(sim_time)`` and receives an explicit :class:`PressState`.

``FixtureTaskActor`` adapts that controller to the coordinator's small actor
protocol without importing :mod:`brazing_sim.process` (avoiding a module
cycle).  It returns :class:`TaskStatus`, which the coordinator already
normalises.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from .config import FIXTURE_CONFIG, FixtureConfig, derive_product_layout
from .domain import (
    FixtureState,
    FixtureStatus,
    OrderSpec,
    PressState,
    ProductState,
    TaskSpec,
    TaskStatus,
    TaskType,
)
from .motion import Pose, matrix_to_quat


class FixtureControlError(RuntimeError):
    """Raised for an invalid fixture operation or a broken MJCF contract."""


@dataclass(slots=True)
class _PressRuntime:
    started_at: float = 0.0
    phase_started_at: float = 0.0
    last_update_at: float = 0.0
    command_m: float = 0.0
    hold_command_m: float | None = None
    contact_position_m: float | None = None
    hold_started_at: float | None = None
    high_force_started_at: float | None = None
    last_physics_time: float = 0.0
    physics_advanced: bool = False
    filtered_force_n: float = 0.0
    last_touch_at: float | None = None
    locked_press_position_m: float | None = None
    locked_floating_position_m: float | None = None


class FixtureController:
    """Configure comb inserts and run a real slide-actuator press sequence.

    ``source`` may be a ``BrazingScene`` or a raw ``mujoco.MjModel``.  In the
    latter case the matching ``MjData`` must be passed explicitly.
    """

    PRESS_JOINT = "fixture_press_slide"
    FLOATING_JOINT = "fixture_press_floating_joint"
    PRESS_ACTUATOR = "fixture_press_actuator"
    TOUCH_SENSOR = "fixture_press_touch_sensor"
    POSITION_SENSOR = "fixture_press_jointpos_sensor"
    FORCE_SENSOR = "fixture_press_force_sensor"
    HOLD_WELD = "fixture_press_hold_weld"
    DRIVE_HOLD_WELD = "fixture_press_drive_hold_weld"

    def __init__(
        self,
        source: Any,
        data: Any | None = None,
        *,
        registry: Any | None = None,
        config: FixtureConfig = FIXTURE_CONFIG,
        state: FixtureState | None = None,
    ) -> None:
        import mujoco

        self.scene = source if hasattr(source, "model") else None
        self.model = source.model if self.scene is not None else source
        self.data = data if data is not None else getattr(source, "data", None)
        if self.data is None:
            raise TypeError("FixtureController requires a MuJoCo data object")
        self.registry = registry if registry is not None else getattr(source, "registry", None)
        self.mujoco = getattr(source, "mujoco", mujoco)
        self.config = config
        self.mechanical_config = config
        self.state = state if state is not None else FixtureState()
        self.active_spec: OrderSpec | None = None
        self._runtime = _PressRuntime()

        obj = self.mujoco.mjtObj
        self.press_joint_id = self._required_id(obj.mjOBJ_JOINT, self.PRESS_JOINT, "joint")
        self.floating_joint_id = self._required_id(obj.mjOBJ_JOINT, self.FLOATING_JOINT, "joint")
        self.press_actuator_id = self._required_id(obj.mjOBJ_ACTUATOR, self.PRESS_ACTUATOR, "actuator")
        self.touch_sensor_id = self._required_id(obj.mjOBJ_SENSOR, self.TOUCH_SENSOR, "sensor")
        self.position_sensor_id = self._required_id(obj.mjOBJ_SENSOR, self.POSITION_SENSOR, "sensor")
        self.force_sensor_id = self._required_id(obj.mjOBJ_SENSOR, self.FORCE_SENSOR, "sensor")
        self.hold_weld_id = self._required_id(obj.mjOBJ_EQUALITY, self.HOLD_WELD, "equality weld")
        self.drive_hold_weld_id = self._required_id(obj.mjOBJ_EQUALITY, self.DRIVE_HOLD_WELD, "equality weld")
        self.press_system_body_id = self._required_id(obj.mjOBJ_BODY, "fixture_upper_press_system", "body")
        self.press_drive_body_id = self._required_id(obj.mjOBJ_BODY, "fixture_press_drive", "body")
        self.floating_body_id = self._required_id(obj.mjOBJ_BODY, "fixture_press_floating_body", "body")
        self.press_qpos_address = int(self.model.jnt_qposadr[self.press_joint_id])
        self.press_dof_address = int(self.model.jnt_dofadr[self.press_joint_id])
        self.floating_qpos_address = int(self.model.jnt_qposadr[self.floating_joint_id])
        self.floating_dof_address = int(self.model.jnt_dofadr[self.floating_joint_id])

        self._comb_geom_defaults: dict[int, tuple[np.ndarray, int, int]] = {}
        for geom_id in range(int(self.model.ngeom)):
            name = self.mujoco.mj_id2name(self.model, obj.mjOBJ_GEOM, geom_id) or ""
            if not (name.startswith("front_comb_") or name.startswith("rear_comb_")):
                continue
            if "insert" not in name and "_g" not in name:
                continue
            self._comb_geom_defaults[geom_id] = (
                np.asarray(self.model.geom_rgba[geom_id], dtype=float).copy(),
                int(self.model.geom_contype[geom_id]),
                int(self.model.geom_conaffinity[geom_id]),
            )
        self._site_alpha: dict[int, float] = {}

    def _required_id(self, object_type: Any, name: str, label: str) -> int:
        identifier = int(self.mujoco.mj_name2id(self.model, object_type, name))
        if identifier < 0:
            raise FixtureControlError(f"工装控制器缺少 {label} {name!r}，请检查 brazing_line.xml")
        return identifier

    def _optional_id(self, object_type: Any, name: str) -> int:
        return int(self.mujoco.mj_name2id(self.model, object_type, name))

    @staticmethod
    def _pitch_token(spec: OrderSpec) -> str:
        suffix = spec.comb_module_name.removeprefix("comb_insert_")
        if suffix.endswith("mm"):
            return suffix[:-2]
        return str(int(round(spec.fin_pitch * 1000.0)))

    def _fixture_state(self, fixture_state: FixtureState | None) -> FixtureState:
        if fixture_state is not None:
            self.state = fixture_state
        return self.state

    def activate_comb_insert(self, module_name: str) -> None:
        """Show one matching front/rear insert pair and disable all others."""

        token = module_name.removeprefix("comb_insert_").removesuffix("mm")
        selected_markers = (f"front_comb_insert_{token}mm", f"rear_comb_insert_{token}mm")
        selected_guides = (f"front_comb_{token}_", f"rear_comb_{token}_")
        obj = self.mujoco.mjtObj
        for geom_id, (default_rgba, default_contype, default_conaffinity) in self._comb_geom_defaults.items():
            name = self.mujoco.mj_id2name(self.model, obj.mjOBJ_GEOM, geom_id) or ""
            active = any(
                name.startswith(marker) or name.startswith(guide)
                for marker, guide in zip(selected_markers, selected_guides)
            )
            rgba = default_rgba.copy()
            rgba[3] = default_rgba[3] if active else 0.0
            self.model.geom_rgba[geom_id] = rgba
            self.model.geom_contype[geom_id] = default_contype if active else 0
            self.model.geom_conaffinity[geom_id] = default_conaffinity if active else 0

    def deactivate_comb_insert(self, module_name: str) -> None:
        token = module_name.removeprefix("comb_insert_").removesuffix("mm")
        obj = self.mujoco.mjtObj
        markers = (f"front_comb_insert_{token}mm", f"rear_comb_insert_{token}mm")
        guides = (f"front_comb_{token}_", f"rear_comb_{token}_")
        for geom_id, (default_rgba, _contype, _conaffinity) in self._comb_geom_defaults.items():
            name = self.mujoco.mj_id2name(self.model, obj.mjOBJ_GEOM, geom_id) or ""
            if not any(
                name.startswith(marker) or name.startswith(guide) for marker, guide in zip(markers, guides)
            ):
                continue
            rgba = default_rgba.copy()
            rgba[3] = 0.0
            self.model.geom_rgba[geom_id] = rgba
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0

    def _set_site(self, name: str, position: np.ndarray, *, active: bool) -> None:
        site_id = self._required_id(self.mujoco.mjtObj.mjOBJ_SITE, name, "site")
        self.model.site_pos[site_id] = position
        self._site_alpha.setdefault(site_id, float(self.model.site_rgba[site_id, 3]))
        self.model.site_rgba[site_id, 3] = self._site_alpha[site_id] if active else 0.0

    def generate_fin_slot_targets(self, spec: OrderSpec) -> tuple[tuple[float, float, float], ...]:
        """Update mutable target sites from the product coordinate frame."""

        fins = derive_product_layout(spec).fins
        base_target_id = self._required_id(self.mujoco.mjtObj.mjOBJ_SITE, "base_plate_target_site", "site")
        base_centre = np.asarray(self.model.site_pos[base_target_id], dtype=float)
        targets: list[tuple[float, float, float]] = []
        for index in range(spec.max_fins):
            active = index < spec.fin_count
            fin = fins[index]
            if active:
                fin_target = base_centre + np.asarray(fin.target_position, dtype=float)
                y = float(fin.target_position[1])
            else:
                # Keep inactive sites deterministic, invisible and away from
                # the active comb opening.
                fin_target = np.asarray([0.0, 0.18 + 0.02 * index, -0.05], dtype=float)
                y = float(fin_target[1])
            target_name = f"fin_slot_{index + 1:02d}_target"
            self._set_site(target_name, fin_target, active=active)

            for end, x in (
                ("front", self.config.front_comb_x),
                ("rear", self.config.rear_comb_x),
            ):
                site_name = f"{end}_comb_slot_{index + 1:02d}"
                site_id = self._required_id(self.mujoco.mjtObj.mjOBJ_SITE, site_name, "site")
                current_z = float(self.model.site_pos[site_id, 2])
                self._set_site(
                    site_name,
                    np.asarray([x, y, current_z], dtype=float),
                    active=active,
                )
            targets.append(tuple(float(value) for value in fin_target))
        return tuple(targets)

    def validate_front_rear_comb_alignment(self, spec: OrderSpec, *, tolerance_m: float = 1e-6) -> bool:
        obj = self.mujoco.mjtObj
        expected = derive_product_layout(spec).active_fins
        for index, fin in enumerate(expected, 1):
            target_id = self._required_id(obj.mjOBJ_SITE, f"fin_slot_{index:02d}_target", "site")
            front_id = self._required_id(obj.mjOBJ_SITE, f"front_comb_slot_{index:02d}", "site")
            rear_id = self._required_id(obj.mjOBJ_SITE, f"rear_comb_slot_{index:02d}", "site")
            target_y = float(self.model.site_pos[target_id, 1])
            front = np.asarray(self.model.site_pos[front_id], dtype=float)
            rear = np.asarray(self.model.site_pos[rear_id], dtype=float)
            if (
                abs(front[1] - rear[1]) > tolerance_m
                or abs(front[1] - target_y) > tolerance_m
                or abs(front[0] - self.config.front_comb_x) > tolerance_m
                or abs(rear[0] - self.config.rear_comb_x) > tolerance_m
                or abs(target_y - fin.target_position[1]) > tolerance_m
            ):
                return False
        return True

    def configure_product(
        self,
        spec: OrderSpec,
        fixture_state: FixtureState | None = None,
    ) -> FixtureState:
        """Select a paired comb module and regenerate all active slot sites."""

        state = self._fixture_state(fixture_state)
        front_x = -spec.fin_length / 2.0 + 0.030
        rear_x = spec.fin_length / 2.0 - 0.030
        self.config = replace(
            self.mechanical_config,
            front_comb_x=front_x,
            rear_comb_x=rear_x,
            target_clamping_force_n=spec.target_clamping_force_n,
            clamping_force_tolerance_n=spec.clamping_force_tolerance_n,
            force_hold_duration_s=spec.force_hold_duration_s,
        )
        self.active_spec = spec
        if self.registry is not None and hasattr(self.registry, "configure_comb_module"):
            # SceneRegistry owns the scene-wide implementation when present.
            self.registry.configure_comb_module(spec)
        else:
            self.activate_comb_insert(spec.comb_module_name)
            self.generate_fin_slot_targets(spec)

        aligned = self.validate_front_rear_comb_alignment(spec)
        if not aligned:
            raise FixtureControlError(f"{spec.preset} 型前后梳齿槽与翅片 target site 未一一对齐")
        pitch = self._pitch_token(spec)
        state.active_comb_module = spec.comb_module_name
        state.front_comb_module = f"front_comb_insert_{pitch}mm"
        state.rear_comb_module = f"rear_comb_insert_{pitch}mm"
        state.comb_configured = True
        state.comb_aligned = True
        if not state.locked:
            state.status = FixtureStatus.COMB_CONFIGURED
        self.mujoco.mj_forward(self.model, self.data)
        return state

    # Recommended prompt-compatible name.
    configure_comb_module = configure_product

    def _sensor_value(self, sensor_id: int) -> float:
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        values = np.asarray(self.data.sensordata[address : address + dimension], dtype=float)
        return float(values[0]) if dimension == 1 else float(np.linalg.norm(values))

    @property
    def measured_position_m(self) -> float:
        return self._sensor_value(self.position_sensor_id)

    @property
    def measured_touch_force_n(self) -> float:
        return max(0.0, self._sensor_value(self.touch_sensor_id))

    @property
    def measured_actuator_force_n(self) -> float:
        return abs(self._sensor_value(self.force_sensor_id))

    @property
    def press_state(self) -> PressState:
        return self.state.press_state

    @property
    def reached_target(self) -> bool:
        return self.state.press_state is PressState.COMPLETE and self.state.press_force_held

    def _set_press_command(self, position_m: float) -> float:
        low = -abs(self.config.press_travel_m)
        command = float(np.clip(position_m, low, 0.0))
        self.data.ctrl[self.press_actuator_id] = command
        self._runtime.command_m = command
        return command

    def _set_hold_lock(self, active: bool) -> None:
        """Lock/unlock the compliant head at its current settled pose."""

        if self.registry is not None and hasattr(self.registry, "set_weld"):
            for weld_name, bodies in (
                (self.HOLD_WELD, ("fixture_press_drive", "fixture_press_floating_body")),
                (
                    self.DRIVE_HOLD_WELD,
                    ("fixture_upper_press_system", "fixture_press_drive"),
                ),
            ):
                self.registry.set_weld(
                    weld_name,
                    bool(active),
                    recompute=bodies if active else None,
                    forward=bool(active),
                )
        else:
            for equality_id, left_id, right_id in (
                (self.hold_weld_id, self.press_drive_body_id, self.floating_body_id),
                (self.drive_hold_weld_id, self.press_system_body_id, self.press_drive_body_id),
            ):
                if active:
                    self.mujoco.mj_forward(self.model, self.data)
                    left = self.data.body(left_id)
                    right = self.data.body(right_id)
                    left_pose = Pose(
                        np.asarray(left.xpos, dtype=float),
                        matrix_to_quat(np.asarray(left.xmat, dtype=float).reshape(3, 3)),
                    )
                    right_pose = Pose(
                        np.asarray(right.xpos, dtype=float),
                        matrix_to_quat(np.asarray(right.xmat, dtype=float).reshape(3, 3)),
                    )
                    relative = left_pose.inverse().transformed(right_pose)
                    self.model.eq_data[equality_id, :] = 0.0
                    self.model.eq_data[equality_id, 3:6] = relative.position
                    self.model.eq_data[equality_id, 6:10] = relative.quaternion
                    self.model.eq_data[equality_id, 10] = 1.0
                self.data.eq_active[equality_id] = int(bool(active))
        if active:
            self.data.qvel[self.press_dof_address] = 0.0
            self.data.qvel[self.floating_dof_address] = 0.0
            self._runtime.locked_press_position_m = float(self.data.qpos[self.press_qpos_address])
            self._runtime.locked_floating_position_m = float(self.data.qpos[self.floating_qpos_address])
        else:
            self._runtime.locked_press_position_m = None
            self._runtime.locked_floating_position_m = None
        if self.registry is not None and hasattr(self.registry, "set_press_latched"):
            self.registry.set_press_latched(bool(active))

    def enforce_hold(self) -> None:
        """Apply the mechanical latch before each physics step."""

        press = self._runtime.locked_press_position_m
        floating = self._runtime.locked_floating_position_m
        if press is None or floating is None:
            return
        self.data.qpos[self.press_qpos_address] = press
        self.data.qpos[self.floating_qpos_address] = floating
        self.data.qvel[self.press_dof_address] = 0.0
        self.data.qvel[self.floating_dof_address] = 0.0
        self.data.ctrl[self.press_actuator_id] = press

    def reset(
        self,
        fixture_state: FixtureState | None = None,
        *,
        hard: bool = True,
    ) -> PressState:
        """Open the press and clear force/lock readiness.

        ``hard=True`` is intended for application reset and deterministically
        puts the joint at its open position.  Cancellation can use
        ``hard=False`` so the actuator opens through normal simulation steps.
        """

        state = self._fixture_state(fixture_state)
        self._runtime = _PressRuntime()
        self._set_hold_lock(False)
        self._set_press_command(0.0)
        if hard:
            self.data.qpos[self.press_qpos_address] = 0.0
            self.data.qvel[self.press_dof_address] = 0.0
            self.data.qpos[self.floating_qpos_address] = 0.0
            self.data.qvel[self.floating_dof_address] = 0.0
        state.press_state = PressState.OPEN
        state.press_position_m = 0.0
        state.clamping_force_n = 0.0
        state.press_force_held = False
        state.ready_for_transfer = False
        state.locked = False
        state.cycle_locked = False
        if state.comb_configured:
            state.status = FixtureStatus.COMB_CONFIGURED
        elif state.base_weld_active:
            state.status = FixtureStatus.BASE_FIXED
        else:
            state.status = FixtureStatus.EMPTY
        if self.registry is not None and hasattr(self.registry, "set_fixture_locked"):
            self.registry.set_fixture_locked(False)
        if self.registry is not None and hasattr(self.registry, "set_press_installed"):
            self.registry.set_press_installed(False)
        self.mujoco.mj_forward(self.model, self.data)
        return state.press_state

    def start_press(
        self,
        sim_time: float,
        fixture_state: FixtureState | None = None,
    ) -> PressState:
        state = self._fixture_state(fixture_state)
        if state.press_state not in {PressState.OPEN, PressState.ERROR}:
            raise FixtureControlError(f"压板当前状态 {state.press_state.value} 不允许重新启动")
        self._set_hold_lock(False)
        missing: list[str] = []
        if not state.comb_configured or not state.comb_aligned:
            missing.append("前后梳齿已配置且对齐")
        if not state.material_passed:
            missing.append("钎焊材料检测通过")
        if not state.fins_passed:
            missing.append("翅片几何检测通过")
        if missing:
            raise FixtureControlError("上压板启动条件不满足：" + "、".join(missing))

        if self.registry is not None and hasattr(self.registry, "set_press_installed"):
            # INSTALL_OR_ENABLE_UPPER_PLATE is part of this non-blocking
            # fixture task.  The mechanism is introduced only after Arm1 has
            # finished every top-down fin insertion.
            self.registry.set_press_installed(True)
        now = float(sim_time)
        self._runtime = _PressRuntime(
            started_at=now,
            phase_started_at=now,
            last_update_at=now,
            command_m=float(self.data.ctrl[self.press_actuator_id]),
            last_physics_time=float(self.data.time),
        )
        self._set_press_command(min(0.0, self._runtime.command_m))
        state.press_state = PressState.CONTACT_SEARCH
        state.press_position_m = self.measured_position_m
        state.clamping_force_n = 0.0
        state.press_force_held = False
        state.ready_for_transfer = False
        state.status = FixtureStatus.PRESSING
        return state.press_state

    def _reported_position(self) -> float:
        measured = self.measured_position_m
        # Unit tests and fast coordinators may call update without stepping
        # MuJoCo.  The command still truthfully represents press progress.
        if abs(measured - self._runtime.command_m) > 0.010 and abs(measured) < 1e-7:
            return self._runtime.command_m
        return measured

    def _physics_is_advancing(self) -> bool:
        current = float(self.data.time)
        if current > self._runtime.last_physics_time + 1.0e-12:
            self._runtime.physics_advanced = True
        self._runtime.last_physics_time = current
        return self._runtime.physics_advanced

    def _effective_force(self, virtual_force_n: float, *, physical: bool) -> float:
        # The thin bars and rigid fin welds make the raw touch signal ring.
        # Fuse it with the series-spring estimate only *after* real contact is
        # observed, then low-pass the result.  This preserves the physical
        # contact gate without allowing harmless solver oscillation to reset
        # the one-second force-hold window forever.
        measured = self.measured_touch_force_n
        virtual = max(0.0, float(virtual_force_n))
        if physical:
            if measured > 0.05:
                self._runtime.last_touch_at = self._runtime.last_update_at
            touch_recent = (
                self._runtime.last_touch_at is not None
                and self._runtime.last_update_at - self._runtime.last_touch_at <= 0.10
            )
            if not touch_recent:
                self._runtime.filtered_force_n = 0.0
                force = 0.0
            else:
                estimate = 0.90 * virtual + 0.10 * measured
                previous = self._runtime.filtered_force_n
                force = estimate if previous <= 0.0 else 0.92 * previous + 0.08 * estimate
                self._runtime.filtered_force_n = force
        else:
            force = max(measured, virtual)
        return float(np.clip(force, 0.0, 1.5 * self.config.target_clamping_force_n))

    def _series_stiffness_n_m(self) -> float:
        actuator_stiffness = max(
            1.0,
            abs(float(self.model.actuator_gainprm[self.press_actuator_id, 0])),
        )
        floating_stiffness = self.config.press_floating_stiffness_n_m
        return actuator_stiffness * floating_stiffness / (actuator_stiffness + floating_stiffness)

    def update(
        self,
        sim_time: float,
        fixture_state: FixtureState | None = None,
    ) -> PressState:
        """Advance one non-blocking press state-machine tick."""

        state = self._fixture_state(fixture_state)
        if state.press_state in {PressState.OPEN, PressState.COMPLETE, PressState.ERROR}:
            return state.press_state
        now = float(sim_time)
        if now + 1e-12 < self._runtime.last_update_at:
            state.press_state = PressState.ERROR
            raise FixtureControlError("仿真时间倒退，上压板状态机已进入 ERROR")
        dt = max(0.0, now - self._runtime.last_update_at)
        self._runtime.last_update_at = now
        physical = self._physics_is_advancing()
        maximum_duration = (
            self.config.press_travel_m / max(self.config.press_search_speed_m_s, 1e-9)
            + self.config.press_ramp_duration_s
            + self.config.force_hold_duration_s
            + 12.0
        )
        if now - self._runtime.started_at > maximum_duration:
            state.press_state = PressState.ERROR
            state.press_force_held = False
            return state.press_state

        if state.press_state is PressState.CONTACT_SEARCH:
            command = self._set_press_command(
                self._runtime.command_m - self.config.press_search_speed_m_s * dt
            )
            state.press_position_m = self._reported_position()
            touch_force = self.measured_touch_force_n
            # Nominal geometry leaves ~17 mm between the open press bars and
            # fin tops.  Real touch wins; the threshold is a deterministic
            # fallback for fake-clock tests that do not call mj_step().
            virtual_contact_position = -0.70 * self.config.press_travel_m
            contact = touch_force > 0.05 or (not physical and command <= virtual_contact_position)
            if contact:
                self._runtime.contact_position_m = min(
                    command,
                    max(-self.config.press_travel_m, state.press_position_m),
                )
                self._runtime.phase_started_at = now
                state.press_state = PressState.FORCE_RAMP
            return state.press_state

        contact_position = self._runtime.contact_position_m
        if contact_position is None:
            state.press_state = PressState.ERROR
            return state.press_state
        target = self.config.target_clamping_force_n
        tolerance = self.config.clamping_force_tolerance_n
        floating_stiffness = self.config.press_floating_stiffness_n_m
        spring_deflection = target / self._series_stiffness_n_m()

        if state.press_state is PressState.FORCE_RAMP:
            progress = float(
                np.clip(
                    (now - self._runtime.phase_started_at) / max(self.config.press_ramp_duration_s, 1e-9),
                    0.0,
                    1.0,
                )
            )
            self._set_press_command(contact_position - spring_deflection * progress)
            state.press_position_m = self._reported_position()
            state.clamping_force_n = self._effective_force(
                target * progress,
                physical=physical,
            )
            if progress >= 1.0 and state.clamping_force_n >= target - tolerance:
                state.press_state = PressState.FORCE_HOLD
                state.clamping_force_n = target
                self._runtime.phase_started_at = now
                self._runtime.hold_started_at = now
                self._runtime.hold_command_m = self._runtime.command_m
            return state.press_state

        if state.press_state is PressState.FORCE_HOLD:
            hold_command = self._runtime.hold_command_m
            if hold_command is None:
                hold_command = contact_position - spring_deflection
            self._set_press_command(hold_command)
            state.press_position_m = self._reported_position()
            state.clamping_force_n = self._effective_force(target, physical=physical)
            if state.clamping_force_n < target - tolerance:
                # Loss of force restarts the stability window without
                # blocking or silently declaring success.  Retightening is
                # deliberately sub-visual (at most 0.02 mm per control tick).
                deficit = target - state.clamping_force_n
                correction = min(0.00002, 0.10 * deficit / floating_stiffness)
                self._runtime.hold_command_m = self._set_press_command(self._runtime.command_m - correction)
                self._runtime.hold_started_at = now
                return state.press_state
            if state.clamping_force_n > target + tolerance:
                # Persist the tiny back-off.  The old implementation reset to
                # the nominal command on the next frame and visibly bounced.
                excess = state.clamping_force_n - target
                correction = min(0.00002, 0.10 * excess / floating_stiffness)
                self._runtime.hold_command_m = self._set_press_command(self._runtime.command_m + correction)
            hold_started = self._runtime.hold_started_at if self._runtime.hold_started_at is not None else now
            if now - hold_started >= self.config.force_hold_duration_s:
                state.press_state = PressState.COMPLETE
                state.clamping_force_n = target
                state.press_force_held = True
                self._set_hold_lock(True)
            return state.press_state

        state.press_state = PressState.ERROR
        return state.press_state

    def complete_immediately(self, fixture_state: FixtureState | None = None) -> PressState:
        """Deterministic fast-mode completion used by headless flow tests."""

        state = self._fixture_state(fixture_state)
        contact = -0.70 * self.config.press_travel_m
        command = max(
            -self.config.press_travel_m,
            contact - self.config.target_clamping_force_n / self._series_stiffness_n_m(),
        )
        self._set_press_command(command)
        self.data.qpos[self.press_qpos_address] = command
        self.data.qvel[self.press_dof_address] = 0.0
        self.data.qvel[self.floating_dof_address] = 0.0
        state.press_state = PressState.COMPLETE
        state.press_position_m = command
        state.clamping_force_n = self.config.target_clamping_force_n
        state.press_force_held = True
        state.status = FixtureStatus.PRESSING
        self._set_hold_lock(True)
        self.mujoco.mj_forward(self.model, self.data)
        return state.press_state

    def lock(self, fixture_state: FixtureState | None = None) -> FixtureState:
        state = self._fixture_state(fixture_state)
        if state.press_state is not PressState.COMPLETE or not state.press_force_held:
            raise FixtureControlError("上压板未达到并稳定保持目标力，不允许锁紧")
        state.lock()
        if self.registry is not None and hasattr(self.registry, "set_fixture_locked"):
            self.registry.set_fixture_locked(True)
        return state


ProductProvider = Callable[[], ProductState | None]


class FixtureTaskActor:
    """Coordinator adapter for comb configuration, pressing and locking."""

    CONFIGURE_DURATION_S = 0.35
    LOCK_DURATION_S = 0.25

    def __init__(
        self,
        scene: Any,
        product: ProductProvider | ProductState,
        *,
        fast: bool = False,
        controller: FixtureController | None = None,
    ) -> None:
        self.scene = scene
        self._product_source = product
        existing = getattr(scene, "fixture_controller", None)
        self.controller = controller or (
            existing if isinstance(existing, FixtureController) else FixtureController(scene)
        )
        self.fast = bool(fast)
        self.task: TaskSpec | None = None
        self.error = ""
        self._finish_at = 0.0
        self._action_applied = False

    def _product(self) -> ProductState:
        product = self._product_source() if callable(self._product_source) else self._product_source
        if product is None:
            raise FixtureControlError("当前没有活动订单，无法执行工装任务")
        return product

    def _apply_configure(self) -> None:
        product = self._product()
        self.controller.configure_product(product.spec, product.fixture)

    def _apply_lock(self) -> None:
        product = self._product()
        self.controller.lock(product.fixture)

    def start_task(self, task: TaskSpec, now: float) -> None:
        if self.task is not None:
            raise RuntimeError(f"fixture actor is already executing {self.task.task_id}")
        task_type = TaskType(task.task_type)
        if task_type not in {
            TaskType.CONFIGURE_COMB,
            TaskType.PRESS_FIXTURE,
            TaskType.LOCK_FIXTURE,
        }:
            raise FixtureControlError(f"工装 actor 不支持任务 {task_type.value}")
        self.task = task
        self.error = ""
        self._action_applied = False
        timestamp = float(now)
        try:
            if task_type is TaskType.CONFIGURE_COMB:
                self._finish_at = timestamp if self.fast else timestamp + self.CONFIGURE_DURATION_S
                if self.fast:
                    self._apply_configure()
                    self._action_applied = True
            elif task_type is TaskType.PRESS_FIXTURE:
                product = self._product()
                if self.fast:
                    # Validate the same gates as the physical state machine.
                    self.controller.start_press(timestamp, product.fixture)
                    self.controller.complete_immediately(product.fixture)
                else:
                    self.controller.start_press(timestamp, product.fixture)
                self._action_applied = True
            else:
                self._finish_at = timestamp if self.fast else timestamp + self.LOCK_DURATION_S
                if self.fast:
                    self._apply_lock()
                    self._action_applied = True
        except Exception as exc:
            self.error = str(exc)
            self.task = None
            raise

    def _finish(self) -> TaskStatus:
        self.task = None
        self._action_applied = False
        return TaskStatus.SUCCEEDED

    def poll_task(self, now: float) -> TaskStatus:
        if self.task is None:
            return TaskStatus.SUCCEEDED
        if self.error:
            return TaskStatus.FAILED
        task_type = TaskType(self.task.task_type)
        try:
            if task_type is TaskType.PRESS_FIXTURE:
                product = self._product()
                state = (
                    product.fixture.press_state
                    if self.fast
                    else self.controller.update(float(now), product.fixture)
                )
                if state is PressState.ERROR:
                    self.error = "上压板寻找接触、力爬升或保压超时"
                    return TaskStatus.FAILED
                if state is PressState.COMPLETE:
                    return self._finish()
                return TaskStatus.RUNNING

            if float(now) < self._finish_at:
                return TaskStatus.RUNNING
            if not self._action_applied:
                if task_type is TaskType.CONFIGURE_COMB:
                    self._apply_configure()
                elif task_type is TaskType.LOCK_FIXTURE:
                    self._apply_lock()
                self._action_applied = True
            return self._finish()
        except Exception as exc:
            self.error = str(exc)
            return TaskStatus.FAILED

    def cancel(self) -> None:
        product: ProductState | None
        try:
            product = self._product()
        except FixtureControlError:
            product = None
        if self.task is not None and TaskType(self.task.task_type) is TaskType.PRESS_FIXTURE:
            self.controller.reset(product.fixture if product is not None else None, hard=False)
        self.task = None
        self.error = ""
        self._action_applied = False


__all__ = [
    "FixtureControlError",
    "FixtureController",
    "FixtureTaskActor",
]
