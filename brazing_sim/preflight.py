"""Startup contract and geometry checks for the flexible brazing fixture.

The checks in this module are deliberately independent of the process
coordinator.  They can be run immediately after MuJoCo compiles the MJCF and
before an order is allowed to start.  Every failure is collected so an XML
author gets one actionable Chinese report instead of fixing names one by one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import (
    DISPENSER_CONFIG,
    FIXTURE_CONFIG,
    ORDER_PRESETS,
    derive_product_layout,
    make_order_spec,
)
from .domain import OrderSpec
from .layout import SHALLOW_U_LAYOUT


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    """One actionable scene-contract or product-geometry problem."""

    object_name: str
    object_type: str
    module: str
    message: str
    xml_file: str = "scenes/production/brazing_line.xml"
    suggestion: str = "请检查 MJCF 中的命名、层级和参数。"

    def format_chinese(self, index: int | None = None) -> str:
        prefix = f"{index}. " if index is not None else ""
        return (
            f"{prefix}[功能模块: {self.module}] {self.message}\n"
            f"   对象: {self.object_name} ({self.object_type})\n"
            f"   XML: {self.xml_file}\n"
            f"   建议: {self.suggestion}"
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "object_name": self.object_name,
            "object_type": self.object_type,
            "module": self.module,
            "message": self.message,
            "xml_file": self.xml_file,
            "suggestion": self.suggestion,
        }


@dataclass(slots=True)
class PreflightReport:
    """Aggregated result returned by :func:`preflight_check`."""

    issues: list[PreflightIssue] = field(default_factory=list)
    checked_presets: tuple[str, ...] = ()
    xml_file: str = "scenes/production/brazing_line.xml"

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def passed(self) -> bool:
        """Readable compatibility alias used by UI and tests."""

        return self.ok

    def add(self, issue: PreflightIssue) -> None:
        # A missing object can be referenced by several product presets.  One
        # copy is enough in the operator-facing report.
        key = (issue.object_name, issue.object_type, issue.module, issue.message)
        if key not in {
            (item.object_name, item.object_type, item.module, item.message) for item in self.issues
        }:
            self.issues.append(issue)

    def format_chinese(self) -> str:
        presets = "/".join(self.checked_presets) if self.checked_presets else "当前订单"
        if self.ok:
            return f"启动预检通过：{presets}，未发现结构或几何错误。"
        lines = [
            f"启动预检失败：{presets}，共 {len(self.issues)} 个问题。",
            "为保护工装和工件，订单流程已禁止启动。",
        ]
        lines.extend(issue.format_chinese(index) for index, issue in enumerate(self.issues, 1))
        return "\n".join(lines)

    def raise_if_failed(self) -> "PreflightReport":
        if not self.ok:
            raise PreflightCheckError(self)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_presets": list(self.checked_presets),
            "xml_file": self.xml_file,
            "issues": [issue.as_dict() for issue in self.issues],
        }


class PreflightCheckError(RuntimeError):
    """Raised when a scene is unsafe to start because preflight failed."""

    def __init__(self, report: PreflightReport) -> None:
        self.report = report
        super().__init__(report.format_chinese())


# Short alias for callers that prefer ``except PreflightError``.
PreflightError = PreflightCheckError


def _source_model(source: Any) -> tuple[Any, Any | None]:
    if hasattr(source, "model"):
        return source.model, source
    return source, None


def _normalise_specs(
    order: OrderSpec | str | Iterable[OrderSpec | str] | None,
) -> tuple[OrderSpec, ...]:
    if order is None:
        return tuple(ORDER_PRESETS[key] for key in sorted(ORDER_PRESETS))
    if isinstance(order, (OrderSpec, str)):
        values: Sequence[OrderSpec | str] = (order,)
    else:
        values = tuple(order)
    result: list[OrderSpec] = []
    for value in values:
        result.append(make_order_spec(value) if isinstance(value, str) else value)
    if not result:
        raise ValueError("preflight order list must not be empty")
    return tuple(result)


def _object_name(model: Any, mujoco: Any, object_type: Any, identifier: int) -> str | None:
    return mujoco.mj_id2name(model, object_type, int(identifier))


def _find(model: Any, mujoco: Any, object_type: Any, name: str) -> int:
    return int(mujoco.mj_name2id(model, object_type, name))


def _add_missing(
    report: PreflightReport,
    model: Any,
    mujoco: Any,
    object_type: Any,
    names: Iterable[str],
    *,
    type_label: str,
    module: str,
) -> None:
    for name in names:
        if _find(model, mujoco, object_type, name) >= 0:
            continue
        report.add(
            PreflightIssue(
                object_name=name,
                object_type=type_label,
                module=module,
                message=f"缺失必需{type_label}。",
                xml_file=report.xml_file,
                suggestion=f"请在 {report.xml_file} 中增加或恢复名为 {name!r} 的{type_label}。",
            )
        )


def _module_pitch_mm(spec: OrderSpec) -> str:
    suffix = spec.comb_module_name.removeprefix("comb_insert_")
    if suffix.endswith("mm"):
        return suffix[:-2]
    return str(int(round(spec.fin_pitch * 1000.0)))


def _check_product_geometry(report: PreflightReport, spec: OrderSpec) -> None:
    layout = derive_product_layout(spec)
    half_x = spec.base_length / 2.0
    half_y = spec.base_width / 2.0
    tol = 1e-9
    for fin in layout.active_fins:
        x, y, _ = fin.target_position
        if abs(x) + spec.fin_length / 2.0 > half_x + tol or abs(y) + spec.fin_thickness / 2.0 > half_y + tol:
            report.add(
                PreflightIssue(
                    object_name=fin.fin_id,
                    object_type="产品槽位",
                    module=f"{spec.preset} 型槽位生成",
                    message="翅片槽位超出基板有效边界。",
                    xml_file=report.xml_file,
                    suggestion="请检查 fin_count、fin_pitch、fin_size 与 base_size。",
                )
            )
    for path in layout.active_paths:
        for endpoint_name, endpoint in (("start", path.local_start), ("end", path.local_end)):
            if (
                abs(endpoint[0]) + path.target_width_m / 2.0 > half_x + tol
                or abs(endpoint[1]) + path.target_width_m / 2.0 > half_y + tol
            ):
                report.add(
                    PreflightIssue(
                        object_name=f"{path.path_id}:{endpoint_name}",
                        object_type="双喷嘴材料线",
                        module=f"{spec.preset} 型材料路径",
                        message="双喷嘴材料线超出基板边界。",
                        xml_file=report.xml_file,
                        suggestion="请检查翅片长度、材料线宽度和 bead_offset_from_slot_center。",
                    )
                )

    if spec.fin_count > 1:
        y_values = [fin.target_position[1] for fin in layout.active_fins]
        pitch_errors = [abs((right - left) - spec.fin_pitch) for left, right in zip(y_values, y_values[1:])]
        if max(pitch_errors, default=0.0) > 1e-9:
            report.add(
                PreflightIssue(
                    object_name=spec.comb_module_name,
                    object_type="梳齿模块",
                    module=f"{spec.preset} 型槽位生成",
                    message="生成的槽位节距与订单节距不一致。",
                    xml_file=report.xml_file,
                )
            )


def _check_comb_geometry(report: PreflightReport, model: Any, mujoco: Any, spec: OrderSpec) -> None:
    """Validate front/rear guide pairs against generated slot centres."""

    pitch_mm = _module_pitch_mm(spec)
    expected = [fin.target_position[1] for fin in derive_product_layout(spec).active_fins]
    tray_id = _find(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, "fixture_tray_geom")
    tray_top = float(model.geom_pos[tray_id, 2] + model.geom_size[tray_id, 2]) if tray_id >= 0 else 0.009
    base_half_length = 0.5 * spec.base_length
    frame_x: dict[str, float] = {}
    pedestal_ids: dict[str, tuple[int, int]] = {}
    for end, sign in (("front", -1.0), ("rear", 1.0)):
        frame_id = _find(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, f"fixture_{end}_comb_frame")
        if frame_id >= 0:
            frame_x[end] = float(model.body_pos[frame_id, 0])
        supports = tuple(
            _find(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, f"{end}_comb_{side}_support")
            for side in ("left", "right")
        )
        top_rail = _find(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, f"{end}_comb_top_rail")
        pedestal_ids[end] = (top_rail,)
        for support in supports:
            if support < 0:
                continue
            bottom = float(model.geom_pos[support, 2] - model.geom_size[support, 2])
            if abs(bottom - tray_top) > 1e-6:
                report.add(
                    PreflightIssue(
                        object_name=f"{end}_comb_support",
                        object_type="梳齿固定基座",
                        module=f"{spec.preset} 型{end}梳齿",
                        message="梳齿基座没有落在托盘顶面。",
                        xml_file=report.xml_file,
                        suggestion="使安装脚底面与 fixture_tray_geom 顶面重合。",
                    )
                )
                break
        if end in frame_x and top_rail >= 0:
            pedestal_world_x = frame_x[end] + float(model.geom_pos[top_rail, 0])
            if sign * pedestal_world_x <= base_half_length:
                report.add(
                    PreflightIssue(
                        object_name=f"{end}_comb_top_rail",
                        object_type="梳齿固定基座",
                        module=f"{spec.preset} 型{end}梳齿",
                        message="梳齿基座侵入基板区域，而不是固定在托盘外侧。",
                        xml_file=report.xml_file,
                    )
                )

    for index, target_y in enumerate(expected, 1):
        centres: dict[str, float] = {}
        for end in ("front", "rear"):
            left_name = f"{end}_comb_{pitch_mm}_g{index:02d}l"
            right_name = f"{end}_comb_{pitch_mm}_g{index:02d}r"
            left = _find(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, left_name)
            right = _find(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, right_name)
            if left < 0 or right < 0:
                # The name-contract check reports the module body/rail; guide
                # names are reported here because a partial module is unsafe.
                for name, identifier in ((left_name, left), (right_name, right)):
                    if identifier < 0:
                        report.add(
                            PreflightIssue(
                                object_name=name,
                                object_type="geom",
                                module=f"{spec.preset} 型{end}梳齿",
                                message="梳齿槽导向指不完整。",
                                xml_file=report.xml_file,
                                suggestion="每个槽位必须同时具有左、右两个导向指。",
                            )
                        )
                continue
            centres[end] = 0.5 * (float(model.geom_pos[left, 1]) + float(model.geom_pos[right, 1]))
            slot_gap = float(
                (model.geom_pos[right, 1] - model.geom_size[right, 1])
                - (model.geom_pos[left, 1] + model.geom_size[left, 1])
            )
            if slot_gap + 1e-9 < spec.fin_thickness:
                report.add(
                    PreflightIssue(
                        object_name=f"{end}_comb_{pitch_mm}_slot_{index:02d}",
                        object_type="梳齿槽",
                        module=f"{spec.preset} 型{end}梳齿",
                        message="梳齿槽宽小于翅片厚度，翅片无法从上方插入。",
                        xml_file=report.xml_file,
                        suggestion="请增大导向指间隙，并保留必要的装配余量。",
                    )
                )
            if abs(centres[end] - target_y) > 1e-6:
                report.add(
                    PreflightIssue(
                        object_name=f"{end}_comb_{pitch_mm}_slot_{index:02d}",
                        object_type="梳齿槽",
                        module=f"{spec.preset} 型{end}梳齿",
                        message=(f"梳齿槽中心 {centres[end]:.6f} m 与订单目标 " f"{target_y:.6f} m 不一致。"),
                        xml_file=report.xml_file,
                        suggestion="请使梳齿导向指中心与产品坐标系槽位 Y 坐标一致。",
                    )
                )
            if end in frame_x:
                guide_min_x = float(model.geom_pos[left, 0] - model.geom_size[left, 0])
                guide_max_x = float(model.geom_pos[left, 0] + model.geom_size[left, 0])
                inner_world_x = frame_x[end] + (guide_max_x if end == "front" else guide_min_x)
                enters_base = (
                    inner_world_x > -base_half_length if end == "front" else inner_world_x < base_half_length
                )
                if not enters_base:
                    report.add(
                        PreflightIssue(
                            object_name=left_name,
                            object_type="悬臂梳齿",
                            module=f"{spec.preset} 型{end}梳齿",
                            message="梳齿没有从托盘基座延伸到基板上空。",
                            xml_file=report.xml_file,
                        )
                    )
                connected = False
                guide_ranges = (
                    (guide_min_x, guide_max_x),
                    (
                        float(model.geom_pos[left, 1] - model.geom_size[left, 1]),
                        float(model.geom_pos[left, 1] + model.geom_size[left, 1]),
                    ),
                    (
                        float(model.geom_pos[left, 2] - model.geom_size[left, 2]),
                        float(model.geom_pos[left, 2] + model.geom_size[left, 2]),
                    ),
                )
                for beam in pedestal_ids.get(end, ()):
                    if beam < 0:
                        continue
                    beam_ranges = tuple(
                        (
                            float(model.geom_pos[beam, axis] - model.geom_size[beam, axis]),
                            float(model.geom_pos[beam, axis] + model.geom_size[beam, axis]),
                        )
                        for axis in range(3)
                    )
                    if all(
                        max(guide_range[0], beam_range[0]) <= min(guide_range[1], beam_range[1]) + 1e-9
                        for guide_range, beam_range in zip(guide_ranges, beam_ranges)
                    ):
                        connected = True
                        break
                if not connected:
                    report.add(
                        PreflightIssue(
                            object_name=left_name,
                            object_type="悬臂梳齿",
                            module=f"{spec.preset} 型{end}梳齿",
                            message="梳齿与托盘上的固定基座断开，视觉上会悬空。",
                            xml_file=report.xml_file,
                            suggestion="使梳齿与低矮顶轨在 X/Y/Z 三向实体重叠。",
                        )
                    )
        if set(centres) == {"front", "rear"} and abs(centres["front"] - centres["rear"]) > 1e-6:
            report.add(
                PreflightIssue(
                    object_name=f"{spec.comb_module_name}:slot_{index:02d}",
                    object_type="前后梳齿槽",
                    module=f"{spec.preset} 型梳齿对齐",
                    message="前后梳齿槽没有一一对齐。",
                    xml_file=report.xml_file,
                    suggestion="请使同一翅片槽的前、后导向指共用同一 Y 坐标。",
                )
            )


def _check_current_sites(
    report: PreflightReport,
    model: Any,
    mujoco: Any,
    spec: OrderSpec,
) -> None:
    expected = [fin.target_position[1] for fin in derive_product_layout(spec).active_fins]
    for index, target_y in enumerate(expected, 1):
        ids = {
            label: _find(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, name)
            for label, name in {
                "target": f"fin_slot_{index:02d}_target",
                "front": f"front_comb_slot_{index:02d}",
                "rear": f"rear_comb_slot_{index:02d}",
            }.items()
        }
        if any(identifier < 0 for identifier in ids.values()):
            continue
        values = {label: float(model.site_pos[identifier, 1]) for label, identifier in ids.items()}
        if any(abs(value - target_y) > 1e-6 for value in values.values()):
            report.add(
                PreflightIssue(
                    object_name=f"slot_{index:02d}",
                    object_type="site 对齐组",
                    module=f"{spec.preset} 型当前槽位",
                    message="槽位 target site 与前后梳齿对齐 site 不共线。",
                    xml_file=report.xml_file,
                    suggestion="请在 configure_comb_module() 中从同一组产品坐标重新生成三个 site。",
                )
            )


def _box_aabb(
    centre: np.ndarray,
    rotation: np.ndarray,
    half_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    extent = np.abs(rotation) @ half_size
    return centre - extent, centre + extent


def _check_raw_material_clearance(
    report: PreflightReport,
    scene: Any | None,
    model: Any,
    mujoco: Any,
    spec: OrderSpec,
) -> None:
    """Reject raw-fin sites that intersect Arm1's physical tool rack.

    The second six-fin bank is only activated by larger products, so checking
    the default A scene is insufficient.  This uses the configured raw sites
    and the rack's contact-enabled box geoms before any robot joint is moved.
    """

    if scene is None or not hasattr(scene, "data"):
        return
    rack_body = _find(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "arm1_tool_rack")
    if rack_body < 0:
        return
    first_geom = int(model.body_geomadr[rack_body])
    geom_count = int(model.body_geomnum[rack_body])
    rack_boxes: list[tuple[str, np.ndarray, np.ndarray]] = []
    for geom_id in range(first_geom, first_geom + geom_count):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if int(model.geom_contype[geom_id]) == 0 or int(model.geom_conaffinity[geom_id]) == 0:
            continue
        centre = np.asarray(scene.data.geom_xpos[geom_id], dtype=float)
        rotation = np.asarray(scene.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        low, high = _box_aabb(centre, rotation, np.asarray(model.geom_size[geom_id, :3], dtype=float))
        rack_boxes.append((model.geom(geom_id).name or str(geom_id), low, high))

    required_clearance = 0.010
    fin_half_size = 0.5 * np.asarray(spec.fin_size, dtype=float)
    for index in range(1, spec.fin_count + 1):
        site_name = f"raw_fin_{index:02d}_site"
        site_id = _find(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        centre = np.asarray(scene.data.site_xpos[site_id], dtype=float)
        rotation = np.asarray(scene.data.site_xmat[site_id], dtype=float).reshape(3, 3)
        fin_low, fin_high = _box_aabb(centre, rotation, fin_half_size)
        for rack_geom_name, rack_low, rack_high in rack_boxes:
            axis_gaps = np.maximum(rack_low - fin_high, fin_low - rack_high)
            # Treat the tool-rack footprint as a keep-out column.  A blank
            # parked directly above the column can still fall or be swept
            # through it during pickup, so vertical separation must not hide
            # an XY intrusion.  Euclidean planar AABB clearance also handles
            # diagonal corner spacing correctly.
            clearance = float(np.linalg.norm(np.maximum(axis_gaps[:2], 0.0)))
            if clearance >= required_clearance:
                continue
            report.add(
                PreflightIssue(
                    object_name=f"{site_name}/{rack_geom_name}",
                    object_type="Table1原料位净距",
                    module=f"{spec.preset} 型Table1原料布局",
                    message=(
                        f"翅片原料位与Arm1工具架净距仅 {clearance * 1000.0:.1f} mm，"
                        f"低于要求的 {required_clearance * 1000.0:.1f} mm。"
                    ),
                    xml_file=report.xml_file,
                    suggestion="请移动工具架或Table1第二列翅片原料位，禁止带穿透启动。",
                )
            )
            break

        # Validate against the complete finished-pallet sweep, not just the
        # narrower black belt that happens to be visible in the viewer.
        lane_left = SHALLOW_U_LAYOUT.output_lane_x - SHALLOW_U_LAYOUT.output_pallet_half_width_m
        lane_clearance = lane_left - float(fin_high[0])
        if lane_clearance < SHALLOW_U_LAYOUT.raw_material_clearance_m:
            report.add(
                PreflightIssue(
                    object_name=f"{site_name}/finished_output_lane",
                    object_type="原料区与成品物流净距",
                    module=f"{spec.preset} 型浅U布局",
                    message=(
                        f"翅片原料位与成品托盘扫掠区净距仅 {lane_clearance * 1000.0:.1f} mm，"
                        f"低于要求的 {SHALLOW_U_LAYOUT.raw_material_clearance_m * 1000.0:.1f} mm。"
                    ),
                    xml_file=report.xml_file,
                    suggestion="请将翅片料仓移到成品输送中心线左侧，禁止原料与成品物流重叠。",
                )
            )


def _check_arm3_camera_mount(
    report: PreflightReport,
    scene: Any | None,
) -> None:
    """Require the Arm3 camera root to stay on the link7 centre axis."""

    if scene is None or not hasattr(scene, "data"):
        return
    link = scene.data.body("arm3_fr3_link7")
    rig = scene.data.body("arm3_camera_rig")
    rotation = np.asarray(link.xmat, dtype=float).reshape(3, 3)
    local = rotation.T @ (np.asarray(rig.xpos, dtype=float) - np.asarray(link.xpos, dtype=float))
    lateral_error = float(np.linalg.norm(local[:2]))
    if lateral_error > 1.0e-6 or abs(float(local[2]) - 0.107) > 1.0e-6:
        report.add(
            PreflightIssue(
                object_name="arm3_camera_rig/arm3_fr3_link7",
                object_type="末端相机安装",
                module="Arm3 同轴腕部相机",
                message=(
                    f"相机根部相对法兰轴线偏移 {lateral_error * 1000.0:.3f} mm，"
                    f"轴向安装距离为 {float(local[2]) * 1000.0:.3f} mm。"
                ),
                xml_file=report.xml_file,
                suggestion="相机根部必须安装在link7局部(0, 0, 0.107)处。",
            )
        )


def _check_shallow_u_station_layout(
    report: PreflightReport,
    scene: Any | None,
) -> None:
    """Verify non-overlapping stations and their authoritative coordinates."""

    if scene is None or not hasattr(scene, "data"):
        return
    expected = {
        "station_s1_anchor": SHALLOW_U_LAYOUT.station_s1_xy,
        "station_s2a_anchor": SHALLOW_U_LAYOUT.station_s2a_xy,
        "station_s2b_anchor": SHALLOW_U_LAYOUT.station_s2b_xy,
        "station_s3_anchor": SHALLOW_U_LAYOUT.station_s3_xy,
        "station_rack_infeed_anchor": SHALLOW_U_LAYOUT.rack_infeed_xy,
    }
    for body_name, xy in expected.items():
        actual = np.asarray(scene.data.body(body_name).xpos[:2], dtype=float)
        error = float(np.linalg.norm(actual - np.asarray(xy, dtype=float)))
        if error <= 1.0e-6:
            continue
        report.add(
            PreflightIssue(
                object_name=body_name,
                object_type="异步工位坐标",
                module="浅U单向物流",
                message=f"工位偏离统一布局配置 {error * 1000.0:.2f} mm。",
                xml_file=report.xml_file,
                suggestion="请同步修改MJCF工位anchor与brazing_sim.layout。",
            )
        )

    table_pairs = (
        ("s1_table", "s2a_table"),
        ("s2a_table", "s2b_table"),
        ("s2b_table", "s3_table"),
    )
    minimum_gap = 0.025
    for left_name, right_name in table_pairs:
        left_id = int(scene.model.geom(left_name).id)
        right_id = int(scene.model.geom(right_name).id)
        left_low, left_high = _box_aabb(
            np.asarray(scene.data.geom_xpos[left_id], dtype=float),
            np.asarray(scene.data.geom_xmat[left_id], dtype=float).reshape(3, 3),
            np.asarray(scene.model.geom_size[left_id, :3], dtype=float),
        )
        right_low, right_high = _box_aabb(
            np.asarray(scene.data.geom_xpos[right_id], dtype=float),
            np.asarray(scene.data.geom_xmat[right_id], dtype=float).reshape(3, 3),
            np.asarray(scene.model.geom_size[right_id, :3], dtype=float),
        )
        horizontal_gaps = np.maximum(
            right_low[:2] - left_high[:2],
            left_low[:2] - right_high[:2],
        )
        clearance = float(np.max(horizontal_gaps))
        if clearance >= minimum_gap:
            continue
        report.add(
            PreflightIssue(
                object_name=f"{left_name}/{right_name}",
                object_type="相邻工位净距",
                module="浅U单向物流",
                message=(
                    f"相邻工位水平净距仅 {clearance * 1000.0:.1f} mm，"
                    f"低于要求的 {minimum_gap * 1000.0:.1f} mm。"
                ),
                xml_file=report.xml_file,
                suggestion="扩大工位中心距并同步重算对应slide joint轴向与行程。",
            )
        )


def _check_nozzles(
    report: PreflightReport,
    model: Any,
    mujoco: Any,
    spec: OrderSpec | None = None,
) -> None:
    left = _find(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, "arm2_left_nozzle_tip_site")
    right = _find(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, "arm2_right_nozzle_tip_site")
    centre = _find(model, mujoco, mujoco.mjtObj.mjOBJ_SITE, "arm2_dispenser_center_tcp")
    if min(left, right, centre) < 0:
        return
    left_pos = np.asarray(model.site_pos[left], dtype=float)
    right_pos = np.asarray(model.site_pos[right], dtype=float)
    centre_pos = np.asarray(model.site_pos[centre], dtype=float)
    spacing = float(np.linalg.norm(right_pos - left_pos))
    expected_spacing = DISPENSER_CONFIG.nozzle_spacing if spec is None else spec.nozzle_spacing
    if spec is not None and abs(spacing - expected_spacing) > 1e-6:
        report.add(
            PreflightIssue(
                object_name="arm2_left_nozzle_tip_site/arm2_right_nozzle_tip_site",
                object_type="site 对",
                module="Arm2 双喷嘴工具",
                message=(f"实际喷嘴间距 {spacing:.6f} m 与配置值 " f"{expected_spacing:.6f} m 不一致。"),
                xml_file=report.xml_file,
                suggestion="请从 DispenserConfig 的对称偏置生成左右 tip site。",
            )
        )
    midpoint = 0.5 * (left_pos + right_pos)
    if float(np.linalg.norm(midpoint - centre_pos)) > 1e-6:
        report.add(
            PreflightIssue(
                object_name="arm2_dispenser_center_tcp",
                object_type="site",
                module="Arm2 双喷嘴工具",
                message="中心 TCP 不在左右喷嘴 tip site 的对称中点。",
                xml_file=report.xml_file,
                suggestion="请将中心 TCP 放在两个喷嘴出口的几何中点。",
            )
        )

    tool_body = _find(
        model,
        mujoco,
        mujoco.mjtObj.mjOBJ_BODY,
        "arm2_dual_brazing_dispenser_tool",
    )
    if tool_body >= 0:
        for site_name, site_id in (
            ("arm2_dispenser_center_tcp", centre),
            ("arm2_left_nozzle_tip_site", left),
            ("arm2_right_nozzle_tip_site", right),
        ):
            if int(model.site_bodyid[site_id]) == tool_body:
                continue
            report.add(
                PreflightIssue(
                    object_name=site_name,
                    object_type="site",
                    module="Arm2 双喷嘴工具",
                    message="喷嘴 site 未挂在双喷嘴工具 body 下。",
                    xml_file=report.xml_file,
                    suggestion="请将中心 TCP 和左右 tip site 放到 arm2_dual_brazing_dispenser_tool 层级内。",
                )
            )


def _check_base_shape(
    report: PreflightReport,
    model: Any,
    mujoco: Any,
    spec: OrderSpec,
) -> None:
    body_id = _find(model, mujoco, mujoco.mjtObj.mjOBJ_BODY, "heatsink_base_plate")
    geom_id = _find(model, mujoco, mujoco.mjtObj.mjOBJ_GEOM, "heatsink_base_plate_geom")
    if body_id < 0 or geom_id < 0:
        return
    geom_count = int(model.body_geomnum[body_id])
    first_geom = int(model.body_geomadr[body_id])
    if geom_count != 1 or first_geom != geom_id:
        report.add(
            PreflightIssue(
                object_name="heatsink_base_plate",
                object_type="body",
                module="纯长方体散热基板",
                message="基板 body 不是单一 geom，仍可能存在安装边、孔或装饰凸起。",
                xml_file=report.xml_file,
                suggestion="请使 heatsink_base_plate 下只保留 heatsink_base_plate_geom。",
            )
        )
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        report.add(
            PreflightIssue(
                object_name="heatsink_base_plate_geom",
                object_type="geom",
                module="纯长方体散热基板",
                message="散热基板不是 box 几何体。",
                xml_file=report.xml_file,
                suggestion="请使用 type='box'，尺寸仅由 OrderSpec.base_size 驱动。",
            )
        )
    expected_half_size = 0.5 * np.asarray(spec.base_size, dtype=float)
    actual_half_size = np.asarray(model.geom_size[geom_id, :3], dtype=float)
    if float(np.max(np.abs(actual_half_size - expected_half_size))) > 1e-6:
        report.add(
            PreflightIssue(
                object_name="heatsink_base_plate_geom",
                object_type="geom",
                module=f"{spec.preset} 型纯长方体散热基板",
                message="基板 geom 尺寸与当前 OrderSpec.base_size 不一致。",
                xml_file=report.xml_file,
                suggestion="请在订单配置后将 geom_size 更新为 base_size / 2。",
            )
        )


def _check_press_wiring(report: PreflightReport, model: Any, mujoco: Any) -> None:
    obj = mujoco.mjtObj
    joint = _find(model, mujoco, obj.mjOBJ_JOINT, "fixture_press_slide")
    actuator = _find(model, mujoco, obj.mjOBJ_ACTUATOR, "fixture_press_actuator")
    if joint >= 0 and actuator >= 0 and int(model.actuator_trnid[actuator, 0]) != joint:
        report.add(
            PreflightIssue(
                object_name="fixture_press_actuator",
                object_type="actuator",
                module="上压板机构",
                message="压紧 actuator 没有驱动 fixture_press_slide。",
                xml_file=report.xml_file,
                suggestion="请将 position actuator 的 joint 设为 fixture_press_slide。",
            )
        )

    expected = {
        "fixture_press_touch_sensor": (
            mujoco.mjtSensor.mjSENS_TOUCH,
            obj.mjOBJ_SITE,
            "fixture_press_touch_site",
        ),
        "fixture_press_jointpos_sensor": (
            mujoco.mjtSensor.mjSENS_JOINTPOS,
            obj.mjOBJ_JOINT,
            "fixture_press_slide",
        ),
        "fixture_press_force_sensor": (
            mujoco.mjtSensor.mjSENS_ACTUATORFRC,
            obj.mjOBJ_ACTUATOR,
            "fixture_press_actuator",
        ),
    }
    for sensor_name, (sensor_type, object_type, target_name) in expected.items():
        sensor_id = _find(model, mujoco, obj.mjOBJ_SENSOR, sensor_name)
        target_id = _find(model, mujoco, object_type, target_name)
        if sensor_id < 0 or target_id < 0:
            continue
        if (
            int(model.sensor_type[sensor_id]) == int(sensor_type)
            and int(model.sensor_objtype[sensor_id]) == int(object_type)
            and int(model.sensor_objid[sensor_id]) == target_id
        ):
            continue
        report.add(
            PreflightIssue(
                object_name=sensor_name,
                object_type="sensor",
                module="上压板机构",
                message=f"压紧传感器类型或绑定对象错误，期望绑定 {target_name}。",
                xml_file=report.xml_file,
                suggestion="请检查 <sensor> 中的 touch、jointpos 和 actuatorfrc 声明。",
            )
        )


def preflight_check(
    source: Any,
    order: OrderSpec | str | Iterable[OrderSpec | str] | None = None,
    *,
    raise_on_error: bool = True,
    xml_file: str | Path = "scenes/production/brazing_line.xml",
    validate_current_sites: bool = False,
) -> PreflightReport:
    """Validate the MJCF contract and all selected order geometries.

    Args:
        source: A ``mujoco.MjModel`` or an object exposing ``.model``.
        order: One order, an iterable of orders, or ``None`` for A/B/C.
        raise_on_error: Raise :class:`PreflightCheckError` after aggregation.
        validate_current_sites: Also check mutable target sites against the
            first selected order.  Enable this *after* scene configuration.
    """

    import mujoco

    model, scene = _source_model(source)
    specs = _normalise_specs(order)
    report = PreflightReport(
        checked_presets=tuple(spec.preset for spec in specs),
        xml_file=str(xml_file),
    )
    obj = mujoco.mjtObj

    _add_missing(
        report,
        model,
        mujoco,
        obj.mjOBJ_BODY,
        (
            "fixture_tray",
            "heatsink_base_plate",
            "fixture_front_comb_frame",
            "fixture_rear_comb_frame",
            "fixture_front_comb_insert",
            "fixture_rear_comb_insert",
            "fixture_upper_press_system",
            "fixture_press_drive",
            "fixture_press_floating_body",
            "fixture_upper_plate",
            "arm2_dual_brazing_dispenser_tool",
            "batch_transfer_base",
            "batch_outfeed_carriage",
            "batch_output_carriage",
            "batch_tray_01",
            "batch_tray_02",
            "batch_tray_03",
            "batch_rack_shelf_0",
            "batch_rack_shelf_1",
            "batch_rack_shelf_2",
            "batch_output_slot_01",
            "batch_output_slot_02",
            "batch_output_slot_03",
            "finished_output_conveyor",
            "finished_output_box",
            "finished_output_gate",
        ),
        type_label="body",
        module="基础工装结构",
    )
    module_bodies: list[str] = []
    module_rails: list[str] = []
    for spec in specs:
        pitch_mm = _module_pitch_mm(spec)
        module_bodies.extend((f"front_comb_insert_{pitch_mm}mm", f"rear_comb_insert_{pitch_mm}mm"))
        module_rails.extend((f"front_comb_insert_{pitch_mm}mm_rail", f"rear_comb_insert_{pitch_mm}mm_rail"))
    _add_missing(
        report,
        model,
        mujoco,
        obj.mjOBJ_BODY,
        module_bodies,
        type_label="body",
        module="可更换前后梳齿模块",
    )
    _add_missing(
        report,
        model,
        mujoco,
        obj.mjOBJ_GEOM,
        (
            "heatsink_base_plate_geom",
            "fixture_front_press_bar",
            "fixture_rear_press_bar",
            "batch_rack_0_lock_pin",
            "batch_rack_1_lock_pin",
            "batch_rack_2_lock_pin",
            "finished_output_belt",
            "finished_output_gate_panel",
            "finished_output_sign",
            *(
                f"batch_tray_{unit:02d}_{part}"
                for unit in range(1, 4)
                for part in (
                    "template_plate",
                    "front_comb_base",
                    "rear_comb_base",
                )
            ),
            *module_rails,
        ),
        type_label="geom",
        module="基板、梳齿与上压板",
    )

    slot_sites = [f"fin_slot_{index:02d}_target" for index in range(1, 13)]
    front_sites = [f"front_comb_slot_{index:02d}" for index in range(1, 13)]
    rear_sites = [f"rear_comb_slot_{index:02d}" for index in range(1, 13)]
    _add_missing(
        report,
        model,
        mujoco,
        obj.mjOBJ_SITE,
        (
            "base_plate_target_site",
            "fixture_tray_grasp_site",
            "fixture_press_touch_site",
            "arm2_dispenser_mount",
            "arm2_dispenser_center_tcp",
            "arm2_left_nozzle_tip_site",
            "arm2_right_nozzle_tip_site",
            *slot_sites,
            *front_sites,
            *rear_sites,
            "batch_transfer_pose",
            "batch_rack_shelf_site_0",
            "batch_rack_shelf_site_1",
            "batch_rack_shelf_site_2",
            "batch_output_slot_01_site",
            "batch_output_slot_02_site",
            "batch_output_slot_03_site",
            "finished_output_inside_site",
        ),
        type_label="site",
        module="产品槽位、压紧与双喷嘴",
    )
    _add_missing(
        report,
        model,
        mujoco,
        obj.mjOBJ_JOINT,
        (
            "fixture_press_slide",
            "fixture_press_floating_joint",
            "batch_outfeed_joint",
            "batch_output_joint",
            "batch_tray_02_index_joint",
            "batch_tray_03_index_joint",
            "finished_output_gate_joint",
        ),
        type_label="joint",
        module="上压板与三层移载机构",
    )
    _add_missing(
        report,
        model,
        mujoco,
        obj.mjOBJ_ACTUATOR,
        (
            "fixture_press_actuator",
            "batch_outfeed_actuator",
            "batch_output_actuator",
            "batch_tray_02_index_actuator",
            "batch_tray_03_index_actuator",
            "finished_output_gate_actuator",
        ),
        type_label="actuator",
        module="上压板与三层移载机构",
    )
    _add_missing(
        report,
        model,
        mujoco,
        obj.mjOBJ_SENSOR,
        (
            "fixture_press_touch_sensor",
            "fixture_press_jointpos_sensor",
            "fixture_press_force_sensor",
            "batch_output_position_sensor",
            "batch_outfeed_position_sensor",
            "finished_output_gate_position_sensor",
        ),
        type_label="sensor",
        module="上压板与三层移载机构",
    )

    equality_names = [
        "tray_fixture_weld",
        "base_tray_weld",
        "fixture_press_hold_weld",
        "fixture_press_drive_hold_weld",
        "arm2_dispenser_tool_weld",
    ]
    equality_names.extend(f"fin_{index:02d}_fixture_weld" for index in range(1, 13))
    equality_names.extend(
        f"slot_{index:02d}_{side}_brazing_path_base_weld"
        for index in range(1, 13)
        for side in ("left", "right")
    )
    equality_names.extend(
        (
            "batch_station_tray_01_weld",
            "batch_indexer_tray_02_weld",
            "batch_indexer_tray_03_weld",
            *(f"batch_carrier_tray_{index:02d}_weld" for index in range(1, 4)),
            *(f"batch_rack_tray_{index:02d}_weld" for index in range(1, 4)),
            *(f"batch_rack_tray_{tray:02d}_shelf_{shelf}_weld" for tray in range(1, 4) for shelf in range(3)),
            *(f"batch_output_tray_{index:02d}_weld" for index in range(1, 4)),
        )
    )
    _add_missing(
        report,
        model,
        mujoco,
        obj.mjOBJ_EQUALITY,
        equality_names,
        type_label="equality weld",
        module="工装锁紧与后续转运",
    )

    for spec in specs:
        _check_product_geometry(report, spec)
        _check_comb_geometry(report, model, mujoco, spec)
    _check_nozzles(report, model, mujoco, specs[0] if validate_current_sites else None)
    _check_press_wiring(report, model, mujoco)
    if validate_current_sites:
        _check_base_shape(report, model, mujoco, specs[0])
        _check_current_sites(report, model, mujoco, specs[0])
        _check_raw_material_clearance(report, scene, model, mujoco, specs[0])
        _check_arm3_camera_mount(report, scene)
        _check_shallow_u_station_layout(report, scene)

    # Ensure the press axis and configured travel agree with the control
    # model.  This catches a renamed joint that still happens to exist with a
    # different type/range.
    press_joint = _find(model, mujoco, obj.mjOBJ_JOINT, "fixture_press_slide")
    if press_joint >= 0:
        if int(model.jnt_type[press_joint]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            report.add(
                PreflightIssue(
                    object_name="fixture_press_slide",
                    object_type="joint",
                    module="上压板机构",
                    message="上压板驱动关节不是 slide joint。",
                    xml_file=report.xml_file,
                )
            )
        axis = np.asarray(model.jnt_axis[press_joint], dtype=float)
        if abs(float(np.dot(axis, np.asarray([0.0, 0.0, 1.0])))) < 1.0 - 1e-6:
            report.add(
                PreflightIssue(
                    object_name="fixture_press_slide",
                    object_type="joint",
                    module="上压板机构",
                    message="上压板 slide joint 未沿工装 Z 轴运动。",
                    xml_file=report.xml_file,
                )
            )
        travel = abs(float(model.jnt_range[press_joint, 1] - model.jnt_range[press_joint, 0]))
        if abs(travel - FIXTURE_CONFIG.press_travel_m) > 1e-6:
            report.add(
                PreflightIssue(
                    object_name="fixture_press_slide",
                    object_type="joint",
                    module="上压板机构",
                    message=(
                        f"MJCF 压板行程 {travel:.6f} m 与配置值 "
                        f"{FIXTURE_CONFIG.press_travel_m:.6f} m 不一致。"
                    ),
                    xml_file=report.xml_file,
                )
            )

    if raise_on_error:
        report.raise_if_failed()
    return report


__all__ = [
    "PreflightCheckError",
    "PreflightError",
    "PreflightIssue",
    "PreflightReport",
    "preflight_check",
]
