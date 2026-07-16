"""Equality-weld quick changers for the Arm1 and Arm2 tool racks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

from .motion import ArmController, Pose, matrix_to_quat, pose_from_site


class ToolLocation(str, Enum):
    ON_RACK = "on_rack"
    ON_ARM = "on_arm"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    body: str
    free_joint: str
    tcp_site: str
    mount_site: str
    rack_site: str
    arm_weld: str
    rack_weld: str


TOOL_SPECS: dict[str, ToolSpec] = {
    "brazing_dispenser": ToolSpec(
        name="brazing_dispenser",
        body="arm2_brazing_dispenser",
        free_joint="arm2_brazing_dispenser_free",
        tcp_site="arm2_brazing_dispenser_tcp",
        mount_site="arm2_brazing_dispenser_mount_site",
        rack_site="arm2_brazing_dispenser_rack_site",
        arm_weld="arm2_toolchange_brazing_dispenser",
        rack_weld="arm2_rack_brazing_dispenser",
    ),
    "tray_transfer": ToolSpec(
        name="tray_transfer",
        body="arm2_tray_transfer",
        free_joint="arm2_tray_transfer_free",
        tcp_site="arm2_tray_transfer_tcp",
        mount_site="arm2_tray_transfer_mount_site",
        rack_site="arm2_tray_transfer_rack_site",
        arm_weld="arm2_toolchange_tray_transfer",
        rack_weld="arm2_rack_tray_transfer",
    ),
}

ARM1_TOOL_SPECS: dict[str, ToolSpec] = {
    "parallel_gripper": ToolSpec(
        name="parallel_gripper",
        body="arm1_parallel_gripper",
        free_joint="arm1_parallel_gripper_free",
        tcp_site="arm1_grasp_tcp",
        mount_site="arm1_parallel_gripper_mount_site",
        rack_site="arm1_parallel_gripper_rack_site",
        arm_weld="arm1_toolchange_parallel_gripper",
        rack_weld="arm1_rack_parallel_gripper",
    ),
    "suction_tool": ToolSpec(
        name="suction_tool",
        body="arm1_suction_tool",
        free_joint="arm1_suction_tool_free",
        tcp_site="arm1_suction_tcp",
        mount_site="arm1_suction_tool_mount_site",
        rack_site="arm1_suction_tool_rack_site",
        arm_weld="arm1_toolchange_suction_tool",
        rack_weld="arm1_rack_suction_tool",
    ),
}


def _body_pose(data: Any, body_id: int) -> Pose:
    body = data.body(body_id)
    return Pose(np.asarray(body.xpos, dtype=float), matrix_to_quat(np.asarray(body.xmat).reshape(3, 3)))


class QuickChangeToolManager:
    """Own exactly-one-of rack/flange weld state for one arm's tools.

    Tools have free roots so they can be parked and subsequently welded to the
    attached FR3 without modifying the upstream robot asset.  Only this class
    changes the tool welds; callers use :meth:`dock`, :meth:`undock` and
    :meth:`reset_to_rack`.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        controller: ArmController | None = None,
        *,
        arm_name: str,
        registry: Mapping[str, ToolSpec] | None = None,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.controller = controller
        self.arm_name = str(arm_name)
        self.registry = dict(registry or TOOL_SPECS)
        self.link7_id = int(model.body(f"{self.arm_name}_fr3_link7").id)
        self.master_site_id = int(model.site(f"{self.arm_name}_attachment_site").id)
        self.body_ids: dict[str, int] = {}
        self.qpos_addresses: dict[str, int] = {}
        self.dof_addresses: dict[str, int] = {}
        self.tcp_site_ids: dict[str, int] = {}
        self.rack_site_ids: dict[str, int] = {}
        self.arm_weld_ids: dict[str, int] = {}
        self.rack_weld_ids: dict[str, int] = {}
        self.parked_qpos: dict[str, np.ndarray] = {}
        for name, spec in self.registry.items():
            try:
                self.body_ids[name] = int(model.body(spec.body).id)
                joint_id = int(model.joint(spec.free_joint).id)
                address = int(model.jnt_qposadr[joint_id])
                self.qpos_addresses[name] = address
                self.dof_addresses[name] = int(model.jnt_dofadr[joint_id])
                self.tcp_site_ids[name] = int(model.site(spec.tcp_site).id)
                self.rack_site_ids[name] = int(model.site(spec.rack_site).id)
                self.arm_weld_ids[name] = int(model.equality(spec.arm_weld).id)
                self.rack_weld_ids[name] = int(model.equality(spec.rack_weld).id)
            except KeyError as exc:
                raise RuntimeError(f"quick-change scene contract is incomplete for {name}: {exc}") from exc
            self.parked_qpos[name] = np.asarray(model.qpos0[address : address + 7], dtype=float).copy()
        self.current_tool: str | None = None
        self.location = {name: ToolLocation.ON_RACK for name in self.registry}
        self.reset_to_rack()

    @property
    def state(self) -> dict[str, str | None]:
        return {
            "current_tool": self.current_tool,
            **{name: location.value for name, location in self.location.items()},
        }

    @property
    def available_tools(self) -> tuple[str, ...]:
        return tuple(self.registry)

    def _require(self, tool: str) -> ToolSpec:
        try:
            return self.registry[tool]
        except KeyError as exc:
            raise ValueError(
                f"unknown {self.arm_name} tool {tool!r}; expected one of {tuple(self.registry)}"
            ) from exc

    def _teleport(self, tool: str, pose: Pose) -> None:
        address = self.qpos_addresses[tool]
        self.data.qpos[address : address + 3] = pose.position
        self.data.qpos[address + 3 : address + 7] = pose.quaternion
        dof = self.dof_addresses[tool]
        self.data.qvel[dof : dof + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def _set_weld(self, equality_id: int, relative_pose: Pose, active: bool) -> None:
        # Weld equality data: anchor[0:3] (unused here), relpos[3:6],
        # relquat[6:10], torque scale[10].
        self.model.eq_data[equality_id, :] = 0.0
        self.model.eq_data[equality_id, 3:6] = relative_pose.position
        self.model.eq_data[equality_id, 6:10] = relative_pose.quaternion
        self.model.eq_data[equality_id, 10] = 1.0
        self.data.eq_active[equality_id] = 1 if active else 0

    def _refresh_controller_tool(self, tool: str | None) -> None:
        if self.controller is None:
            return
        if tool is None:
            self.controller.set_tool_transform(None)
            return
        flange = pose_from_site(self.data, self.master_site_id)
        tcp = pose_from_site(self.data, self.tcp_site_ids[tool])
        self.controller.set_tool_transform(flange.inverse().transformed(tcp))

    def dock(self, tool: str) -> None:
        """Snap a rack tool onto this arm and switch its equality welds."""

        self._require(tool)
        if self.current_tool == tool:
            return
        if self.current_tool is not None:
            raise RuntimeError(f"return {self.current_tool} before docking {tool}")
        if self.location[tool] is not ToolLocation.ON_RACK:
            raise RuntimeError(f"tool {tool} is not available on the rack")

        self.data.eq_active[self.rack_weld_ids[tool]] = 0
        flange_pose = pose_from_site(self.data, self.master_site_id)
        self._teleport(tool, flange_pose)
        link_pose = _body_pose(self.data, self.link7_id)
        tool_pose = _body_pose(self.data, self.body_ids[tool])
        relative = link_pose.inverse().transformed(tool_pose)
        self._set_weld(self.arm_weld_ids[tool], relative, True)
        self.location[tool] = ToolLocation.ON_ARM
        self.current_tool = tool
        self.mujoco.mj_forward(self.model, self.data)
        self._refresh_controller_tool(tool)

    def undock(self, tool: str | None = None) -> None:
        """Return the mounted tool to its exact qpos0 rack pose."""

        selected = self.current_tool if tool is None else tool
        if selected is None:
            return
        self._require(selected)
        if self.current_tool != selected:
            raise RuntimeError(f"cannot return {selected}: current tool is {self.current_tool}")
        self.data.eq_active[self.arm_weld_ids[selected]] = 0
        qpos = self.parked_qpos[selected]
        self._teleport(selected, Pose(qpos[:3], qpos[3:7]))
        # The rack weld's compiler-generated relpose is unchanged by docking.
        self.data.eq_active[self.rack_weld_ids[selected]] = 1
        self.location[selected] = ToolLocation.ON_RACK
        self.current_tool = None
        self._refresh_controller_tool(None)
        self.mujoco.mj_forward(self.model, self.data)

    def change_tool(self, tool: str) -> None:
        self._require(tool)
        if self.current_tool == tool:
            return
        if self.current_tool is not None:
            self.undock(self.current_tool)
        self.dock(tool)

    @property
    def current_body(self) -> str | None:
        if self.current_tool is None:
            return None
        return self.registry[self.current_tool].body

    def tool_transform(self, tool: str) -> Pose:
        """Return the flange-to-TCP transform for a named tool."""

        self._require(tool)
        mount = pose_from_site(self.data, int(self.model.site(self.registry[tool].mount_site).id))
        tcp = pose_from_site(self.data, self.tcp_site_ids[tool])
        return mount.inverse().transformed(tcp)

    def tcp_for_flange(self, flange_pose: Pose, tool: str | None = None) -> Pose:
        """Convert a desired flange pose to the TCP pose used by the IK layer."""

        selected = self.current_tool if tool is None else tool
        if selected is None:
            return flange_pose
        return flange_pose.transformed(self.tool_transform(selected))

    def sync_mounted(self) -> None:
        """Keep the mounted free-body tool rigid during kinematic joint playback."""

        if self.current_tool is None:
            return
        flange_pose = pose_from_site(self.data, self.master_site_id)
        self._teleport(self.current_tool, flange_pose)

    def reset_to_rack(self) -> None:
        """Emergency-safe reset that releases every flange weld."""

        for name in self.registry:
            self.data.eq_active[self.arm_weld_ids[name]] = 0
            self.data.eq_active[self.rack_weld_ids[name]] = 0
            address = self.qpos_addresses[name]
            qpos = self.parked_qpos[name]
            self.data.qpos[address : address + 7] = qpos
            dof = self.dof_addresses[name]
            self.data.qvel[dof : dof + 6] = 0.0
            self.data.eq_active[self.rack_weld_ids[name]] = 1
            self.location[name] = ToolLocation.ON_RACK
        self.current_tool = None
        self._refresh_controller_tool(None)
        self.mujoco.mj_forward(self.model, self.data)

    def tcp_pose(self) -> Pose:
        if self.current_tool is None:
            return pose_from_site(self.data, self.master_site_id)
        return pose_from_site(self.data, self.tcp_site_ids[self.current_tool])

    def rack_pose(self, tool: str) -> Pose:
        self._require(tool)
        return pose_from_site(self.data, self.rack_site_ids[tool])

    def change_poses(self, tool: str, hover_m: float = 0.12) -> tuple[Pose, Pose, Pose]:
        """Return approach, dock and retreat flange poses for a tool change.

        The rack tools are parked adapter-up, while an FR3 flange uses local +Z
        as the working direction.  A 180 degree X rotation provides a top-down
        docking pose.
        """

        dock_position = self.rack_pose(tool).position
        dock = Pose(dock_position, np.asarray([0.0, 1.0, 0.0, 0.0]))
        hover = Pose(dock_position + np.asarray([0.0, 0.0, float(hover_m)]), dock.quaternion)
        return hover, dock, hover


class Arm1ToolManager(QuickChangeToolManager):
    def __init__(self, model: Any, data: Any, controller: ArmController | None = None) -> None:
        super().__init__(
            model,
            data,
            controller,
            arm_name="arm1",
            registry=ARM1_TOOL_SPECS,
        )


class Arm2ToolManager(QuickChangeToolManager):
    def __init__(
        self,
        model: Any,
        data: Any,
        controller: ArmController | None = None,
        registry: Mapping[str, ToolSpec] | None = None,
    ) -> None:
        super().__init__(
            model,
            data,
            controller,
            arm_name="arm2",
            registry=registry or TOOL_SPECS,
        )


# Concise compatibility alias for existing Arm2 callers.
ToolManager = Arm2ToolManager


__all__ = [
    "ARM1_TOOL_SPECS",
    "Arm1ToolManager",
    "Arm2ToolManager",
    "QuickChangeToolManager",
    "TOOL_SPECS",
    "ToolLocation",
    "ToolManager",
    "ToolSpec",
]
