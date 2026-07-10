"""Object manager: component knowledge base, order presets, component/screw
state machines, simulated camera perception and rack scattering.

Perception note: `perceive()` reads ground-truth poses from the simulation.
Swap its body for a real vision pipeline (camera -> pose estimate) without
touching any caller: the interface is (position, yaw).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from skill_library import yaw_from_mat


# ---------------------------------------------------------------------------
# Type-level component knowledge (flexibility: knowledge belongs to the TYPE;
# instance poses are perceived at run time).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentSpec:
    type_name: str
    grip_width: float
    half_height: float
    grasp_depth: float
    grasp_yaw_in_object: float
    yaw_symmetry_rad: float
    place_drop: float = 0.002
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
        release_dwell_s=1.2,
    ),
    "breaker": ComponentSpec(
        type_name="breaker",
        grip_width=0.028,
        half_height=0.025,
        grasp_depth=0.014,
        grasp_yaw_in_object=math.pi / 2.0,
        yaw_symmetry_rad=math.pi,
    ),
}

COMPONENT_INSTANCES: dict[str, str] = {
    "relay_1": "relay",
    "relay_2": "relay",
    "terminal_1": "terminal",
    "button_1": "button",
    "breaker_1": "breaker",
}

# Table2 positioning fixture: one dedicated staging slot per component type.
TYPE_STAGING_SITE: dict[str, str] = {
    "relay": "staging_relay",
    "terminal": "staging_terminal",
    "button": "staging_button",
    "breaker": "staging_breaker",
}

# Orders in the project-plan JSON dialect (section 6).
ORDER_PRESETS: dict[str, dict[str, Any]] = {
    "A": {
        "order_id": "A_basic",
        "components": [
            {"type": "relay", "instance": "relay_1", "target_slot": "slot_1"},
            {"type": "terminal", "instance": "terminal_1", "target_slot": "slot_5"},
        ],
        "screws": ["screw_slot_1_a", "screw_slot_1_b"],
        "inspection_points": ["inspect_slot_1"],
    },
    "B": {
        "order_id": "B_complex",
        "components": [
            {"type": "relay", "instance": "relay_2", "target_slot": "slot_2"},
            {"type": "breaker", "instance": "breaker_1", "target_slot": "slot_4"},
            {"type": "terminal", "instance": "terminal_1", "target_slot": "slot_6"},
        ],
        "screws": ["screw_slot_2_a", "screw_slot_2_b", "screw_slot_4_a", "screw_slot_4_b"],
        "inspection_points": ["inspect_slot_2", "inspect_slot_4"],
    },
    "C": {
        "order_id": "C_rush",
        "components": [
            {"type": "relay", "target_slot": "slot_3"},
        ],
        "screws": ["screw_slot_3_a", "screw_slot_3_b"],
        "inspection_points": ["inspect_slot_3"],
    },
}

SCREW_LOOSE_RGBA = (0.8, 0.15, 0.1, 1.0)
SCREW_TIGHT_RGBA = (0.1, 0.8, 0.2, 1.0)

# Component lifecycle: raw (scattered on Table1) -> staged (in the type slot on
# Table2) -> placed (dropped into the board slot) -> assembled (press-fitted)
# -> pass / fail (inspection verdict).
COMPONENT_STATES = ("raw", "staged", "placed", "assembled", "pass", "fail")

# Feed rack scatter region (Table1 top face, world frame).
RACK_CENTER = np.asarray([-1.02, 0.35], dtype=float)
RACK_HALF_XY = np.asarray([0.10, 0.08], dtype=float)
RACK_TOP_Z = 0.10
SCATTER_MIN_SEPARATION = 0.075


class ObjectManager:
    """Centralized component/screw state + simulated perception."""

    def __init__(self, model: Any, data: Any) -> None:
        self.model = model
        self.data = data
        self.component_body_ids = {name: int(model.body(name).id) for name in COMPONENT_INSTANCES}
        self.component_joint_qposadr: dict[str, int] = {}
        for name in COMPONENT_INSTANCES:
            jid = int(model.joint(f"{name}_free").id)
            self.component_joint_qposadr[name] = int(model.jnt_qposadr[jid])
        self.screw_geom_ids: dict[str, int] = {}
        self.screw_site_ids: dict[str, int] = {}
        for i in range(model.ngeom):
            gname = model.geom(i).name
            if gname and gname.startswith("g_screw_"):
                self.screw_geom_ids[gname[len("g_"):]] = int(i)
        for i in range(model.nsite):
            sname = model.site(i).name
            if sname and sname.startswith("screw_"):
                self.screw_site_ids[sname] = int(i)

        self.component_status: dict[str, str] = {name: "raw" for name in COMPONENT_INSTANCES}
        self.screw_state: dict[str, str] = {}

    # -- state machine ------------------------------------------------------
    def set_status(self, instance: str, status: str) -> None:
        if status not in COMPONENT_STATES:
            raise ValueError(f"unknown component status: {status}")
        self.component_status[instance] = status

    def status_of(self, instance: str) -> str:
        return self.component_status.get(instance, "raw")

    def reset_statuses(self) -> None:
        for name in self.component_status:
            self.component_status[name] = "raw"

    # -- screws ---------------------------------------------------------------
    def set_screw_state(self, screw: str, state: str) -> None:
        self.screw_state[screw] = state
        gid = self.screw_geom_ids.get(screw)
        if gid is not None:
            self.model.geom_rgba[gid] = SCREW_TIGHT_RGBA if state == "tightened" else SCREW_LOOSE_RGBA

    # -- simulated camera perception ------------------------------------------
    def perceive(self, instance: str) -> tuple[np.ndarray, float]:
        """Simulated camera: reads the ground-truth pose from the physics
        state. Replace with a real vision estimate to close the sim-to-real
        gap; every skill only consumes (position, yaw)."""
        body_id = self.component_body_ids[instance]
        pos = np.array(self.data.body(body_id).xpos, copy=True)
        yaw = yaw_from_mat(np.asarray(self.data.body(body_id).xmat, dtype=float))
        return pos, yaw

    def spec_of(self, instance: str) -> ComponentSpec:
        return COMPONENT_SPECS[COMPONENT_INSTANCES[instance]]

    # -- pose writes ------------------------------------------------------------
    def teleport(self, instance: str, pos_xyz: np.ndarray, yaw: float) -> None:
        """Write a free-joint pose directly (used by scatter + fixture snap)."""
        import mujoco  # local import keeps module import light

        adr = self.component_joint_qposadr[instance]
        self.data.qpos[adr:adr + 3] = np.asarray(pos_xyz, dtype=float)
        half = 0.5 * float(yaw)
        self.data.qpos[adr + 3:adr + 7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
        jid = int(self.model.joint(f"{instance}_free").id)
        dofadr = int(self.model.jnt_dofadr[jid])
        self.data.qvel[dofadr:dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def randomize_rack_poses(self, seed: Optional[int] = None) -> dict[str, tuple[float, float, float]]:
        """Scatter all raw components across Table1: uniform position inside
        the rack area, fully random yaw, rejection-sampled minimum separation.
        Deterministic for a given seed (flexibility demo: change the seed, no
        code or teaching changes needed)."""
        rng = np.random.default_rng(seed)
        for _attempt in range(50):
            placed: list[np.ndarray] = []
            layout: dict[str, tuple[np.ndarray, float]] = {}
            ok = True
            for instance in sorted(COMPONENT_INSTANCES):
                for _ in range(300):
                    xy = RACK_CENTER + (rng.random(2) * 2.0 - 1.0) * RACK_HALF_XY
                    if all(float(np.linalg.norm(xy - p)) >= SCATTER_MIN_SEPARATION for p in placed):
                        break
                else:
                    ok = False
                    break
                placed.append(xy)
                layout[instance] = (xy, float(rng.uniform(-math.pi, math.pi)))
            if ok:
                break
        else:
            raise RuntimeError("could not scatter components without overlap")

        result: dict[str, tuple[float, float, float]] = {}
        for instance, (xy, yaw) in layout.items():
            spec = self.spec_of(instance)
            z = RACK_TOP_Z + float(spec.half_height) + 0.001
            self.teleport(instance, np.asarray([xy[0], xy[1], z]), yaw)
            result[instance] = (float(xy[0]), float(xy[1]), yaw)
        return result

    def perturb_component(self, instance: str, dx: float = 0.008, dy: float = 0.006, dyaw_deg: float = 10.0) -> None:
        """Misalignment fault injection: nudge an already-placed component so
        the next inspection fails on position error and triggers re-pressing."""
        pos, yaw = self.perceive(instance)
        self.teleport(
            instance,
            pos + np.asarray([float(dx), float(dy), 0.0]),
            yaw + math.radians(float(dyaw_deg)),
        )
