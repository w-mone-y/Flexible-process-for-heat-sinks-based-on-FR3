"""Arm2 automatic tool changer (design doc sections 2-5).

Arm2 carries no permanent gripper: its flange has a quick-change MASTER plate
(`arm2_tool_changer_master` / `arm2_tool_mount_site`). Two tools live on
`arm2_tool_rack`:

  gripper_press  grip + press-fit combo tool (arm2_grasp_tcp / arm2_press_tcp)
  screwdriver    automatic electric screwdriver (arm2_screwdriver_tcp)

Mount / park states are simulated with equality welds (plan simplification):
exactly one of {tool<->hand, tool<->rack} is active per tool at any time. On
dock the tool is snapped to the exact mated pose (adapter on the master plate,
tool axes == hand axes so the combo-tool fingers open along the same axis the
legacy gripper did) and the arm's `tool_offsets` are re-read from the live TCP
sites, so all skill pose math keeps working with the current tool geometry.

State dict (design doc section 5):
    {"current_tool": None|"gripper_press"|"screwdriver",
     "gripper_press": "on_rack"|"on_arm",
     "screwdriver":   "on_rack"|"on_arm"}
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from arm_controller import ArmController
from skill_library import (
    SkillStep,
    compute_tool_ee_pose,
    mat_to_quat,
    pose_cmd,
    top_down_ee_quat,
)

# Hand axes must coincide with the parked tool axes at dock time so the combo
# tool fingers open along hand-y exactly like the legacy FR3 gripper. With the
# tools parked adapter-up (quat 0 1 0 0) this fixes the EE yaw:
TOOL_DOCK_YAW = -math.pi / 2.0
TOOL_HOVER = 0.12                 # vertical clearance above the dock pose, m

TOOL_NAMES = ("gripper_press", "screwdriver")
TOOL_BODY = {
    "gripper_press": "arm2_gripper_press_tool",
    "screwdriver": "arm2_screwdriver_tool",
}
TOOL_RACK_SITE = {
    "gripper_press": "arm2_gripper_press_rack_site",
    "screwdriver": "arm2_screwdriver_rack_site",
}
TOOL_ARM_WELD = {
    "gripper_press": "arm2_toolchange_gripper_press",
    "screwdriver": "arm2_toolchange_screwdriver",
}
TOOL_RACK_WELD = {
    "gripper_press": "arm2_rack_gripper_press",
    "screwdriver": "arm2_rack_screwdriver",
}
# TCP sites per tool -> names used in ArmController.tool_offsets. The skills
# consume "grasp" implicitly (grasp TCP == attachment site by construction),
# "press" (press_steps) and "screwdriver" (screw_steps).
TOOL_TCP_SITES = {
    "gripper_press": {"grasp": "arm2_grasp_tcp", "press": "arm2_press_tcp"},
    "screwdriver": {"screwdriver": "arm2_screwdriver_tcp"},
}


class Arm2ToolManager:
    """Owns the quick-change weld bookkeeping + tool-change skill steps."""

    def __init__(self, model: Any, data: Any, arm: ArmController) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.arm = arm

        def site_id(name: str) -> int:
            try:
                return int(model.site(name).id)
            except KeyError as exc:
                raise RuntimeError(f"tool changer XML incomplete: missing site {name}") from exc

        def body_id(name: str) -> int:
            try:
                return int(model.body(name).id)
            except KeyError as exc:
                raise RuntimeError(f"tool changer XML incomplete: missing body {name}") from exc

        def eq_id(name: str) -> int:
            try:
                return int(model.equality(name).id)
            except KeyError as exc:
                raise RuntimeError(f"tool changer XML incomplete: missing weld {name}") from exc

        self.master_site_id = site_id("arm2_tool_mount_site")
        self.rack_body_id = body_id("arm2_tool_rack")
        self.tool_body_ids = {t: body_id(TOOL_BODY[t]) for t in TOOL_NAMES}
        self.rack_site_ids = {t: site_id(TOOL_RACK_SITE[t]) for t in TOOL_NAMES}
        self.arm_weld_ids = {t: eq_id(TOOL_ARM_WELD[t]) for t in TOOL_NAMES}
        self.rack_weld_ids = {t: eq_id(TOOL_RACK_WELD[t]) for t in TOOL_NAMES}
        self.tcp_site_ids = {
            t: {tcp: site_id(sname) for tcp, sname in TOOL_TCP_SITES[t].items()} for t in TOOL_NAMES
        }
        self.tool_qposadr = {}
        self.tool_dofadr = {}
        for t in TOOL_NAMES:
            jid = int(model.joint(f"{TOOL_BODY[t]}_free").id)
            self.tool_qposadr[t] = int(model.jnt_qposadr[jid])
            self.tool_dofadr[t] = int(model.jnt_dofadr[jid])

        # Fixed offsets read at the home configuration.
        ee_pos = np.asarray(data.site(arm.site_id).xpos, dtype=float)
        ee_mat = np.asarray(data.site(arm.site_id).xmat, dtype=float).reshape(3, 3)
        master_pos = np.asarray(data.site(self.master_site_id).xpos, dtype=float)
        # Master-plate dock point expressed in the EE-site frame (constant).
        self.master_offset_ee = ee_mat.T @ (master_pos - ee_pos)
        hand_pos = np.asarray(data.body(arm.hand_body_id).xpos, dtype=float)
        hand_mat = np.asarray(data.body(arm.hand_body_id).xmat, dtype=float).reshape(3, 3)
        # Dock point in the hand frame: mated tool origin sits here.
        self.mount_offset_hand = hand_mat.T @ (master_pos - hand_pos)

        # Parked poses (world) + park relpose in the rack frame, from qpos0.
        rack_pos = np.asarray(data.body(self.rack_body_id).xpos, dtype=float)
        rack_mat = np.asarray(data.body(self.rack_body_id).xmat, dtype=float).reshape(3, 3)
        self.parked_pose: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.parked_rel: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for t in TOOL_NAMES:
            tpos = np.array(data.body(self.tool_body_ids[t]).xpos, copy=True)
            tmat = np.asarray(data.body(self.tool_body_ids[t]).xmat, dtype=float).reshape(3, 3)
            self.parked_pose[t] = (tpos, mat_to_quat(tmat))
            self.parked_rel[t] = (rack_mat.T @ (tpos - rack_pos), mat_to_quat(rack_mat.T @ tmat))

        self.state: dict[str, Optional[str]] = {
            "current_tool": None,
            "gripper_press": "on_rack",
            "screwdriver": "on_rack",
        }

    # -- low-level helpers ----------------------------------------------------
    def _teleport_tool(self, tool: str, pos: np.ndarray, quat: np.ndarray) -> None:
        adr = self.tool_qposadr[tool]
        self.data.qpos[adr:adr + 3] = np.asarray(pos, dtype=float)
        self.data.qpos[adr + 3:adr + 7] = np.asarray(quat, dtype=float)
        dof = self.tool_dofadr[tool]
        self.data.qvel[dof:dof + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def _set_weld(self, eq_id: int, rel_pos: np.ndarray, rel_quat: np.ndarray, active: bool) -> None:
        self.model.eq_data[eq_id, :] = 0.0
        self.model.eq_data[eq_id, 3:6] = rel_pos
        self.model.eq_data[eq_id, 6:10] = rel_quat
        self.model.eq_data[eq_id, 10] = 1.0
        self.data.eq_active[eq_id] = 1 if active else 0

    # -- dock / undock (called from skill on_action) ---------------------------
    def dock(self, tool: str) -> None:
        if tool not in TOOL_NAMES:
            raise ValueError(f"unknown tool: {tool}")
        if self.state["current_tool"] is not None:
            raise RuntimeError(f"cannot mount {tool}: {self.state['current_tool']} still on the flange")
        hand_pos = np.asarray(self.data.body(self.arm.hand_body_id).xpos, dtype=float)
        hand_mat = np.asarray(self.data.body(self.arm.hand_body_id).xmat, dtype=float).reshape(3, 3)
        # Snap the tool onto the master plate: tool axes == hand axes.
        self._teleport_tool(tool, hand_pos + hand_mat @ self.mount_offset_hand, mat_to_quat(hand_mat))
        self.data.eq_active[self.rack_weld_ids[tool]] = 0
        self._set_weld(
            self.arm_weld_ids[tool],
            self.mount_offset_hand,
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            active=True,
        )
        self.state[tool] = "on_arm"
        self.state["current_tool"] = tool
        self._refresh_tool_offsets(tool)
        print(f"[Arm2] Tool {tool} mounted", flush=True)

    def undock(self, tool: str) -> None:
        if self.state["current_tool"] != tool:
            raise RuntimeError(f"cannot return {tool}: current tool is {self.state['current_tool']}")
        self.data.eq_active[self.arm_weld_ids[tool]] = 0
        pos, quat = self.parked_pose[tool]
        self._teleport_tool(tool, pos, quat)
        rel_pos, rel_quat = self.parked_rel[tool]
        self._set_weld(self.rack_weld_ids[tool], rel_pos, rel_quat, active=True)
        self.state[tool] = "on_rack"
        self.state["current_tool"] = None
        self.arm.tool_offsets.clear()
        print(f"[Arm2] Tool {tool} returned to rack", flush=True)

    def _refresh_tool_offsets(self, tool: str) -> None:
        """Re-read TCP offsets (EE-site frame) from the live mated sites so the
        skill pose math always uses the actually mounted tool geometry."""
        ee_pos = np.asarray(self.data.site(self.arm.site_id).xpos, dtype=float)
        ee_mat = np.asarray(self.data.site(self.arm.site_id).xmat, dtype=float).reshape(3, 3)
        self.arm.tool_offsets.clear()
        for tcp, sid in self.tcp_site_ids[tool].items():
            tcp_pos = np.asarray(self.data.site(sid).xpos, dtype=float)
            self.arm.tool_offsets[tcp] = ee_mat.T @ (tcp_pos - ee_pos)

    # -- tool-change skill steps -----------------------------------------------
    def _dock_ee_pose(self, tool: str) -> tuple[np.ndarray, np.ndarray]:
        rack_pos = np.asarray(self.data.site(self.rack_site_ids[tool]).xpos, dtype=float)
        quat = top_down_ee_quat(TOOL_DOCK_YAW)
        return compute_tool_ee_pose(rack_pos, quat, self.master_offset_ee), quat

    def change_tool_steps(self, tool: str) -> list[SkillStep]:
        """Ensure `tool` is mounted (design doc section 5): no-op if already
        mounted, otherwise auto-return the wrong tool first, then fetch."""
        if tool not in TOOL_NAMES:
            raise ValueError(f"unknown tool: {tool}")
        if self.state["current_tool"] == tool:
            return []
        steps: list[SkillStep] = []
        if self.state["current_tool"] is not None:
            steps.extend(self.return_tool_steps())
        steps.extend(self._fetch_steps(tool))
        return steps

    def get_tool_steps(self, tool: str) -> list[SkillStep]:
        """approach rack -> descend onto the adapter -> dock (weld swap) -> lift."""
        if tool not in TOOL_NAMES:
            raise ValueError(f"unknown tool: {tool}")
        if self.state["current_tool"] == tool:
            return []
        if self.state["current_tool"] is not None:
            raise RuntimeError("return the current tool before fetching another one")
        return self._fetch_steps(tool)

    def _fetch_steps(self, tool: str) -> list[SkillStep]:
        print(f"[Arm2] Request tool: {tool}", flush=True)
        dock_ee, quat = self._dock_ee_pose(tool)
        hover = dock_ee + np.asarray([0.0, 0.0, TOOL_HOVER])
        return [
            SkillStep("tool_rack_approach", pose=pose_cmd(hover, quat)),
            SkillStep("tool_dock_descend", pose=pose_cmd(dock_ee, quat), pos_tol=0.003, ori_tol=0.03),
            SkillStep("tool_dock", pose=pose_cmd(dock_ee, quat), action="tool_dock", action_arg=tool, dwell_s=0.3, pos_tol=0.003),
            SkillStep("tool_lift", pose=pose_cmd(hover, quat), dwell_s=0.1),
        ]

    def reset_to_rack(self) -> None:
        """Emergency reset: park both tools back on the rack (stop / fault)."""
        mounted = self.state["current_tool"]
        if mounted is not None:
            self.data.eq_active[self.arm_weld_ids[mounted]] = 0
            pos, quat = self.parked_pose[mounted]
            self._teleport_tool(mounted, pos, quat)
            rel_pos, rel_quat = self.parked_rel[mounted]
            self._set_weld(self.rack_weld_ids[mounted], rel_pos, rel_quat, active=True)
        for tool in TOOL_NAMES:
            self.state[tool] = "on_rack"
            self.data.eq_active[self.arm_weld_ids[tool]] = 0
            self.data.eq_active[self.rack_weld_ids[tool]] = 1
        self.state["current_tool"] = None
        self.arm.tool_offsets.clear()

    def return_tool_steps(self) -> list[SkillStep]:
        """Carry the mounted tool back over its rack slot, undock, lift away."""
        tool = self.state["current_tool"]
        if tool is None:
            return []
        print(f"[Arm2] Return tool {tool}", flush=True)
        dock_ee, quat = self._dock_ee_pose(tool)
        hover = dock_ee + np.asarray([0.0, 0.0, TOOL_HOVER])
        return [
            SkillStep("tool_rack_approach", pose=pose_cmd(hover, quat)),
            SkillStep("tool_park_descend", pose=pose_cmd(dock_ee, quat), pos_tol=0.003, ori_tol=0.03),
            SkillStep("tool_undock", pose=pose_cmd(dock_ee, quat), action="tool_undock", action_arg=tool, dwell_s=0.3, pos_tol=0.003),
            SkillStep("tool_lift", pose=pose_cmd(hover, quat), dwell_s=0.1),
        ]
