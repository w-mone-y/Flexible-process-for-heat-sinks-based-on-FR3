"""Optional Qt control panel and independent Arm3 camera window."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Any, Iterable

from .api import get_bytes, get_json, post_json
from .fault_catalog import MANUAL_FAULT_CATALOG
from .planning.task_models import task_detail_label_zh, task_status_label_zh, task_type_label_zh

TASK_GRAPH_NODE_SIZE = (190.0, 72.0)
PLANNING_TAB_TITLES = (
    "运行总览",
    "柔性总览",
    "订单规划",
    "任务图 / 调度",
    "异步流水工位",
    "实时甘特图",
    "产品工程图规划",
    "资源与区域",
    "故障与恢复规划",
    "批次与物流",
    "指标与实验",
)


@dataclass(frozen=True, slots=True)
class UiSegmentAction:
    label_zh: str
    segment: str


@dataclass(frozen=True, slots=True)
class LineUiProfile:
    profile_id: str
    window_title: str
    tab_titles: tuple[str, ...]
    segment_actions: tuple[UiSegmentAction, ...]
    station_titles: dict[str, str]


_V1_UI_PROFILE = LineUiProfile(
    profile_id="V1_STANDARD",
    window_title="低压配电柜散热组件钎焊 MuJoCo 仿真",
    tab_titles=PLANNING_TAB_TITLES,
    segment_actions=(
        UiSegmentAction("单独运行取放", "pick_place"),
        UiSegmentAction("检测1", "inspection_1"),
        UiSegmentAction("Arm2运动", "arm2_motion"),
        UiSegmentAction("翅片安装", "fin_assembly"),
        UiSegmentAction("检测2", "inspection_2"),
        UiSegmentAction("压紧/进炉/返回", "furnace_cycle"),
    ),
    station_titles={
        "S1_BASE_LOADING": "S1 基板装载",
        "S2A_DISPENSING": "S2A 钎料涂覆",
        "S2B_MATERIAL_INSPECTION": "S2B 材料检测",
        "S3_FIN_ASSEMBLY": "S3 翅片装配/压紧",
        "RACK_INFEED": "炉前料架入口",
    },
)

_V2_UI_PROFILE = LineUiProfile(
    profile_id="V2_DUAL_INSTALL",
    window_title="V2 双安装支路柔性钎焊 MuJoCo 仿真",
    tab_titles=PLANNING_TAB_TITLES,
    segment_actions=(
        UiSegmentAction("基板上料", "v2_base_loading"),
        UiSegmentAction("钎料涂覆", "v2_dispensing"),
        UiSegmentAction("焊料检测", "v2_material_inspection"),
        UiSegmentAction("Arm1 安装支路 A", "v2_install_a"),
        UiSegmentAction("Arm3 安装支路 B", "v2_install_b"),
        UiSegmentAction("双支路并行安装", "v2_parallel_install"),
        UiSegmentAction("Y 合流与焊前检测", "v2_merge_inspection"),
        UiSegmentAction("三层炉批", "v2_furnace_batch"),
        UiSegmentAction("炉后检测与交付", "v2_post_braze_delivery"),
    ),
    # Keys must match the ``station_id`` values V2 actually puts on its tasks,
    # otherwise the station filter silently matches nothing.  The three numbered
    # buffers were exactly that case: tasks carry the unnumbered
    # ``FURNACE_BUFFER``, so selecting "炉前缓存 1" filtered out every task.
    station_titles={
        "S1_BASE_LOADING": "S1 基板上料",
        "S2A_DISPENSING": "S2A 钎料涂覆",
        "S2B_MATERIAL_INSPECTION": "S2B 焊料检测与分流",
        "INSTALL_BRANCH_PENDING": "安装支路待分配",
        "S3A_ARM1_INSTALL": "S3A Arm1 翅片安装",
        "S3B_ARM3_INSTALL": "S3B Arm3 翅片安装",
        "Y_MERGE_SHARED": "Y 形合流单占用区",
        "S4_PRE_BRAZE_INSPECTION": "S4 共享焊前检测",
        "FURNACE_BUFFER": "炉前缓存位",
        "FURNACE_FRONT": "炉前门装载",
        "FURNACE": "三层贯通炉",
        "FURNACE_REAR": "炉后门卸载",
        "POST_BRAZE_SCAN": "焊后固定视觉检测",
        "FINISHED_OUTPUT": "成品出口",
    },
)


def line_ui_profile(profile_id: str | None) -> LineUiProfile:
    """Return one immutable UI description without importing Qt."""

    normalized = str(profile_id or "V1_STANDARD").strip().upper()
    if normalized in {"V2", "V2_DUAL_INSTALL"}:
        return _V2_UI_PROFILE
    return _V1_UI_PROFILE


def unique_order_id(preferred: str, unavailable: Iterable[str] = ()) -> str:
    """Return a stable UI order id that does not collide with prior submissions.

    The order runtime intentionally retains completed orders for metrics and task-
    graph history, so reusing ``UI_ORDER_001`` is a duplicate even after its
    physical product has left the workcell.  Keep a human-readable numeric suffix
    instead of hiding that lifecycle rule behind a random UUID.
    """

    candidate = str(preferred).strip() or "UI_ORDER_001"
    occupied = {str(value).strip() for value in unavailable if str(value).strip()}
    if candidate not in occupied:
        return candidate
    match = re.fullmatch(r"(.*?)(\d+)", candidate)
    if match is None:
        prefix, number, width = f"{candidate}_", 1, 3
    else:
        prefix, digits = match.groups()
        number, width = int(digits) + 1, len(digits)
    while True:
        candidate = f"{prefix}{number:0{width}d}"
        if candidate not in occupied:
            return candidate
        number += 1


def manual_review_popup_state(state: dict[str, Any]) -> dict[str, str] | None:
    """Select and format the current nonblocking manual-review popup."""

    notices = [item for item in state.get("manual_review_notices", ()) if isinstance(item, dict)]
    active = [item for item in notices if item.get("status") == "MANUAL_REVIEW"]
    succeeded = [item for item in notices if item.get("status") == "SUCCEEDED"]
    if active:
        item = active[-1]
    elif succeeded:
        item = succeeded[-1]
    else:
        return None
    recovery_id = str(item.get("recovery_id", ""))
    status = str(item.get("status", ""))
    message = str(item.get("message", ""))
    if status == "MANUAL_REVIEW" and item.get("complete_at") is not None:
        remaining = max(0.0, float(item["complete_at"]) - float(state.get("sim_time", 0.0)))
        message = f"{message}\n预计剩余 {remaining:.1f} 秒"
    return {"recovery_id": recovery_id, "status": status, "message": message}


def _place_task_graph_node(item: Any, x: float, y: float) -> None:
    """Move a node whose rectangle geometry is local to the item origin."""

    item.setPos(float(x), float(y))


def _paint_task_graph_node_text(
    painter: Any,
    rect: Any,
    title: str,
    detail: str,
    status: str,
) -> None:
    """Paint labels in the node's own paint pass for reliable macOS rendering."""

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPainter, QPen

    painter.save()
    try:
        painter.setOpacity(1.0)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        font = painter.font()
        font.setPixelSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255, 255), 1))
        metrics = painter.fontMetrics()
        top_text = metrics.elidedText(str(title), Qt.ElideRight, int(rect.width() - 16.0))
        top = rect.adjusted(8.0, 3.0, -6.0, -rect.height() + 28.0)
        painter.drawText(top, int(Qt.AlignLeft | Qt.AlignVCenter), top_text)
        font.setPixelSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(QColor(235, 242, 248, 255), 1))
        metrics = painter.fontMetrics()
        detail_text = metrics.elidedText(str(detail), Qt.ElideRight, int(rect.width() - 16.0))
        middle = rect.adjusted(8.0, 27.0, -6.0, -rect.height() + 50.0)
        painter.drawText(middle, int(Qt.AlignLeft | Qt.AlignVCenter), detail_text)
        painter.setPen(QPen(QColor(255, 255, 255, 255), 1))
        status_text = metrics.elidedText(f"状态：{status}", Qt.ElideRight, int(rect.width() - 16.0))
        bottom = rect.adjusted(8.0, 49.0, -6.0, -3.0)
        painter.drawText(bottom, int(Qt.AlignLeft | Qt.AlignVCenter), status_text)
    finally:
        painter.restore()


def _base_url(value: Any) -> str:
    if isinstance(value, str):
        return value.rstrip("/")
    host = getattr(value, "host", "127.0.0.1")
    port = int(getattr(value, "port", 8765))
    return f"http://{host}:{port}"


def run_ui_client(args_or_url: Any = "http://127.0.0.1:8765") -> int:
    """Run the compact line panel.

    PySide6 is imported lazily so headless simulation and unit tests do not
    require a display stack.
    """

    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QColor, QBrush, QPainter, QPen, QPixmap
        from PySide6.QtSvg import QSvgGenerator
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QFileDialog,
            QFrame,
            QGraphicsScene,
            QGraphicsRectItem,
            QGraphicsView,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSpinBox,
            QTabWidget,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print("PySide6 is not installed; run with --headless or install brazing-sim[ui].", file=sys.stderr)
        return 2

    base_url = _base_url(args_or_url)
    selected_profile = line_ui_profile(getattr(args_or_url, "line_profile", None))

    class TemperaturePlot(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.values: list[float] = []
            self.setMinimumHeight(110)

        def add(self, value: float) -> None:
            self.values.append(float(value))
            self.values = self.values[-180:]
            self.update()

        def paintEvent(self, event: Any) -> None:  # type: ignore[override]
            del event
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#111820"))
            if len(self.values) < 2:
                return
            painter.setPen(QPen(QColor("#ff9d3d"), 2))
            width = max(1, self.width() - 12)
            height = max(1, self.height() - 12)
            peak = max(650.0, max(self.values))
            points = []
            for index, value in enumerate(self.values):
                x = 6 + width * index / max(1, len(self.values) - 1)
                y = 6 + height * (1.0 - max(0.0, value) / peak)
                points.append((int(x), int(y)))
            for left, right in zip(points, points[1:]):
                painter.drawLine(left[0], left[1], right[0], right[1])

    class CameraWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(
                "V2 双检测相机（Arm3 / 焊后固定）"
                if selected_profile.profile_id == "V2_DUAL_INSTALL"
                else "Arm3 inspection camera"
            )
            layout = QVBoxLayout(self)
            self.status = QLabel("camera starting")
            self.status.setAlignment(Qt.AlignCenter)
            self.image = QLabel()
            self.image.setAlignment(Qt.AlignCenter)
            self.image.setMinimumSize(0, 0)
            self.image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.image.setStyleSheet("background:#05070a;border:2px solid #3b4654")
            layout.addWidget(self.status)
            layout.addWidget(self.image, 1)
            self.pixmap = QPixmap()
            self.last_frame_time = 0.0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(200)

        def refresh(self) -> None:
            try:
                state = get_json(base_url + "/camera/status", timeout=0.35)
                frame_time = float(state.get("camera_frame_time", 0.0))
                if frame_time > self.last_frame_time:
                    payload = get_bytes(base_url + "/camera.ppm", timeout=0.35)
                    pixmap = QPixmap()
                    if not pixmap.loadFromData(payload, "PPM"):
                        raise ValueError("invalid camera frame")
                    self.pixmap = pixmap
                    self.last_frame_time = frame_time
                    self.show_pixmap()
                active = bool(state.get("camera_active", False))
                self.status.setText(
                    ("● INSPECTING  " if active else "○ STANDBY  ") + str(state.get("camera_status", ""))
                )
                border = "#23d18b" if active else "#3b4654"
                self.image.setStyleSheet(f"background:#05070a;border:2px solid {border}")
            except Exception as exc:
                self.status.setText(f"camera waiting: {exc}")

        def show_pixmap(self) -> None:
            if not self.pixmap.isNull():
                self.image.setPixmap(
                    self.pixmap.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

        def resizeEvent(self, event: Any) -> None:  # type: ignore[override]
            self.show_pixmap()
            super().resizeEvent(event)

    class TaskNodeItem(QGraphicsRectItem):
        """One graph node with text painted atomically above its fill."""

        def __init__(self, title: str, detail: str, status: str, fill: str) -> None:
            width, height = TASK_GRAPH_NODE_SIZE
            super().__init__(0.0, 0.0, width, height)
            self.title = str(title)
            self.detail = str(detail)
            self.status = str(status)
            self.setPen(QPen(QColor("#8b949e"), 1))
            self.setBrush(QBrush(QColor(fill)))

        def set_content(self, title: str, detail: str, status: str, fill: str) -> None:
            self.title = str(title)
            self.detail = str(detail)
            self.status = str(status)
            self.setBrush(QBrush(QColor(fill)))
            self.update()

        def paint(self, painter: Any, option: Any, widget: Any = None) -> None:  # type: ignore[override]
            super().paint(painter, option, widget)
            _paint_task_graph_node_text(
                painter,
                self.rect(),
                self.title,
                self.detail,
                self.status,
            )

    class TaskGraphView(QGraphicsView):
        COLORS = {
            "PENDING": "#54606f",
            "READY": "#2f81f7",
            "RESERVED": "#a371f7",
            "RUNNING": "#d29922",
            "SUCCEEDED": "#238636",
            "FAILED": "#da3633",
            "BLOCKED": "#6e3b3b",
            "CANCELLED": "#484f58",
            "RETRY_WAIT": "#bf8700",
        }

        def __init__(self) -> None:
            super().__init__()
            self.canvas = QGraphicsScene(self)
            self.setScene(self.canvas)
            self.setRenderHint(QPainter.Antialiasing)
            self.setMinimumHeight(260)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.signature: tuple[Any, ...] | None = None
            self.nodes: dict[str, TaskNodeItem] = {}

        def set_tasks(self, tasks: list[dict[str, Any]]) -> None:
            # Rebuild only when graph topology changes.  Status/progress changes
            # update the existing items in-place, eliminating the visible flash
            # caused by clearing the whole scene four times per second.
            signature = tuple(
                (
                    task.get("task_id"),
                    task.get("task_type"),
                    tuple(task.get("predecessors", ())),
                )
                for task in tasks
            )
            if signature == self.signature:
                for task in tasks:
                    task_id = str(task.get("task_id"))
                    node = self.nodes.get(task_id)
                    if node is None:
                        continue
                    status = str(task.get("status", "PENDING"))
                    progress = float(task.get("progress", 0.0))
                    status_zh = str(task.get("status_zh") or task_status_label_zh(status))
                    if status == "RUNNING":
                        status_zh = f"{status_zh} · {progress * 100.0:.0f}%"
                    node.set_content(
                        str(task.get("display_name_zh") or task_type_label_zh(task.get("task_type", ""))),
                        str(task.get("display_detail_zh") or task_detail_label_zh(task)),
                        status_zh,
                        self.COLORS.get(status, "#54606f"),
                    )
                return
            self.signature = signature
            self.canvas.clear()
            self.nodes.clear()
            if not tasks:
                self.canvas.addText("当前没有任务图。请先在“订单规划”中预览或加入订单。")
                return
            by_id = {str(task.get("task_id")): task for task in tasks}
            levels: dict[str, int] = {}
            for task in tasks:
                predecessors = [str(value) for value in task.get("predecessors", ())]
                levels[str(task.get("task_id"))] = (
                    0
                    if not predecessors
                    else 1 + max((levels.get(value, 0) for value in predecessors), default=0)
                )
            rows: dict[int, int] = {}
            positions: dict[str, tuple[float, float]] = {}
            for task in tasks:
                task_id = str(task.get("task_id"))
                level = levels.get(task_id, 0)
                row = rows.get(level, 0)
                rows[level] = row + 1
                x, y = level * 220.0, row * 98.0
                positions[task_id] = (x, y)
                status = str(task.get("status", "PENDING"))
                node_width, node_height = TASK_GRAPH_NODE_SIZE
                title_zh = str(task.get("display_name_zh") or task_type_label_zh(task.get("task_type", "")))
                detail_zh = str(task.get("display_detail_zh") or task_detail_label_zh(task))
                status_zh = str(task.get("status_zh") or task_status_label_zh(status))
                if status == "RUNNING":
                    status_zh = f"{status_zh} · {float(task.get('progress', 0.0)) * 100.0:.0f}%"
                rect = TaskNodeItem(
                    title_zh,
                    detail_zh,
                    status_zh,
                    self.COLORS.get(status, "#54606f"),
                )
                self.canvas.addItem(rect)
                self.nodes[task_id] = rect
                _place_task_graph_node(rect, x, y)
                rect.setZValue(1.0)
                rect.setToolTip(
                    f"{task_id}\n资源: {task.get('assigned_resource') or task.get('eligible_resources')}\n"
                    f"区域: {task.get('required_zones')}\n错误: {task.get('failure_reason') or '-'}"
                )
            for task in tasks:
                target = str(task.get("task_id"))
                if target not in positions:
                    continue
                tx, ty = positions[target]
                for predecessor in task.get("predecessors", ()):
                    source = str(predecessor)
                    if source not in positions or source not in by_id:
                        continue
                    sx, sy = positions[source]
                    edge = self.canvas.addLine(
                        sx + node_width,
                        sy + node_height / 2.0,
                        tx,
                        ty + node_height / 2.0,
                        QPen(QColor("#8b949e"), 1),
                    )
                    edge.setZValue(-1.0)
            self.canvas.setSceneRect(self.canvas.itemsBoundingRect().adjusted(-20, -20, 20, 20))

    class EngineeringDrawing(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.plan: dict[str, Any] = {}
            self.setMinimumHeight(300)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        def set_plan(self, plan: dict[str, Any]) -> None:
            self.plan = dict(plan)
            self.update()

        def paintEvent(self, event: Any) -> None:  # type: ignore[override]
            del event
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor("#10161d"))
            painter.setPen(QPen(QColor("#d0d7de"), 1))
            if not self.plan:
                painter.drawText(self.rect(), Qt.AlignCenter, "请先预览订单以生成产品工程示意。")
                return
            base = self.plan.get("base_size_m", [0.36, 0.22, 0.008])
            fins = self.plan.get("fin_targets", [])
            paths = self.plan.get("brazing_paths", [])
            left, top = 70.0, 70.0
            width = min(self.width() * 0.60, 650.0)
            height = width * float(base[1]) / max(1e-9, float(base[0]))
            painter.setBrush(QBrush(QColor("#66727f")))
            painter.drawRect(int(left), int(top), int(width), int(height))
            for fin in fins:
                y_m = float(fin.get("position", [0, 0, 0])[1])
                y = top + height * (0.5 - y_m / float(base[1]))
                painter.setPen(QPen(QColor("#f0f6fc"), 3))
                painter.drawLine(int(left + 25), int(y), int(left + width - 25), int(y))
            painter.setPen(QPen(QColor("#f2cc60"), 2))
            for path in paths:
                start = path.get("start", [0, 0, 0])
                end = path.get("end", [0, 0, 0])
                x1 = left + width * (float(start[0]) / float(base[0]) + 0.5)
                x2 = left + width * (float(end[0]) / float(base[0]) + 0.5)
                y = top + height * (0.5 - float(start[1]) / float(base[1]))
                painter.drawLine(int(x1), int(y), int(x2), int(y))
            painter.setPen(QPen(QColor("#58a6ff"), 1))
            painter.drawLine(int(left), int(top - 22), int(left + width), int(top - 22))
            painter.drawText(
                int(left + width / 2 - 70), int(top - 28), f"基板长度 {1000*float(base[0]):.0f} mm"
            )
            details = [
                f"产品：{self.plan.get('product_id', '-')} / {self.plan.get('preset', '-')}型",
                f"基板：{1000*float(base[0]):.0f} × {1000*float(base[1]):.0f} × {1000*float(base[2]):.1f} mm",
                f"翅片：{self.plan.get('fin_count', 0)}片，节距 {1000*float(self.plan.get('fin_pitch_m', 0)):.1f} mm",
                f"钎料：{self.plan.get('path_count', 0)}条，边距 {1000*float(self.plan.get('path_margin_m', 0)):.1f} mm",
                f"喷嘴中心距：{1000*float(self.plan.get('nozzle_spacing_m', 0)):.1f} mm",
                f"梳齿：{self.plan.get('comb_module', '-')}",
                f"压紧力：{float(self.plan.get('clamping_force_n', 0)):.1f} N",
                f"料架层：{[int(v)+1 for v in self.plan.get('rack_layers', [])]}",
                "注：本图为仿真规划示意，不是生产级CAD图。",
            ]
            painter.setPen(QPen(QColor("#d0d7de"), 1))
            x = left + width + 45
            for index, line in enumerate(details):
                painter.drawText(int(x), int(top + 24 * index), line)

    class ControlPanel(QWidget):
        @staticmethod
        def _scroll_page(page: Any) -> Any:
            page.setMinimumSize(0, 0)
            page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            scroll = QScrollArea()
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setMinimumSize(0, 0)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setWidget(page)
            return scroll

        @staticmethod
        def _configure_table(table: Any) -> None:
            table.setMinimumSize(0, 0)
            table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            table.horizontalHeader().setStretchLastSection(True)

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(selected_profile.window_title)
            self.setMinimumSize(720, 480)
            shell = QVBoxLayout(self)
            self.tabs = QTabWidget()
            self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            shell.addWidget(self.tabs)
            overview = QWidget()
            root = QVBoxLayout(overview)
            self.tabs.addTab(self._scroll_page(overview), "运行总览")
            self.current_plan: dict[str, Any] = {}
            self.current_recovery_id = ""
            self.latest_state: dict[str, Any] = {}
            self._fault_target_signature: tuple[str, ...] = ()
            self._submitted_order_ids: set[str] = set()
            self._table_signatures: dict[int, tuple[tuple[str, ...], ...]] = {}
            self.segment_buttons: dict[str, Any] = {}
            self._manual_review_dialog: Any | None = None
            self._manual_review_recovery_id = ""
            self._manual_review_success_seen: set[str] = set()

            controls = QGroupBox("单段流程控制")
            row = QHBoxLayout(controls)
            for action in selected_profile.segment_actions:
                button = self._button(
                    row,
                    action.label_zh,
                    "/segment",
                    {"segment": action.segment},
                )
                self.segment_buttons[action.segment] = button
                if selected_profile.profile_id == "V2_DUAL_INSTALL":
                    button.setEnabled(False)
                    button.setToolTip("等待对应 V2 物理 actor 接通")
            self._button(row, "Stop", "/stop", {})
            self._button(row, "Continue", "/continue", {})
            self._button(row, "Reset", "/reset", {})
            root.addWidget(controls)

            if selected_profile.profile_id == "V1_STANDARD":
                batch_controls = QGroupBox("三层炉内料架")
                batch_row = QHBoxLayout(batch_controls)
                self._button(
                    batch_row,
                    "运行三层批次",
                    "/batch",
                    {"preset": "A", "layers": 3},
                )
                self._button(
                    batch_row,
                    "单独运行直线入炉",
                    "/segment",
                    {"segment": "rack_transfer"},
                )
                root.addWidget(batch_controls)

            speed_controls = QGroupBox("仿真速度")
            speed_row = QHBoxLayout(speed_controls)
            self._button(speed_row, "减速 ÷2", "/speed", {"action": "decelerate"})
            self.speed = QLabel("目标速度: 1× | 实际: --")
            self.speed.setAlignment(Qt.AlignCenter)
            speed_row.addWidget(self.speed, 1)
            self._button(speed_row, "加速 ×2", "/speed", {"action": "accelerate"})
            root.addWidget(speed_controls)

            status_group = QGroupBox("产线状态")
            grid = QGridLayout(status_group)
            self.stage = QLabel("stage: IDLE")
            self.order = QLabel("order: -")
            self.fixture = QLabel("fixture: -")
            self.furnace = QLabel("furnace: -")
            self.result = QLabel("result: -")
            grid.addWidget(self.order, 0, 0)
            grid.addWidget(self.stage, 0, 1)
            grid.addWidget(self.fixture, 1, 0)
            grid.addWidget(self.furnace, 1, 1)
            grid.addWidget(self.result, 2, 0, 1, 2)
            root.addWidget(status_group)

            arms_group = QGroupBox("三臂任务")
            arms_layout = QVBoxLayout(arms_group)
            self.arms = {name: QLabel(f"{name}: idle") for name in ("arm1", "arm2", "arm3")}
            for label in self.arms.values():
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                arms_layout.addWidget(label)
            root.addWidget(arms_group)

            batch_group = QGroupBox("批次、料架与移载")
            batch_layout = QVBoxLayout(batch_group)
            self.batch_status = QLabel("batch: -")
            self.rack_status = QLabel("rack: EMPTY | EMPTY | EMPTY")
            self.transfer_status = QLabel("transfer: IDLE")
            for label in (self.batch_status, self.rack_status, self.transfer_status):
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                batch_layout.addWidget(label)
            root.addWidget(batch_group)

            progress_group = QGroupBox("进度、检测与 KPI")
            progress_layout = QVBoxLayout(progress_group)
            self.progress = QLabel("fins 0/5 | paths 0/10")
            self.arm2_tool = QLabel("Arm2 固定工具：双喷嘴焊料枪")
            self.conveyor = QLabel("conveyor: IDLE")
            self.inspection = QLabel("inspection: -")
            self.kpi = QLabel("KPI: -")
            for label in (
                self.progress,
                self.arm2_tool,
                self.conveyor,
                self.inspection,
                self.kpi,
            ):
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                progress_layout.addWidget(label)
            self.plot = TemperaturePlot()
            progress_layout.addWidget(self.plot)
            root.addWidget(progress_group, 1)

            self._build_planning_tabs()

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(250)

        def _build_flexibility_tab(self) -> None:
            """Six-dimension flexibility evidence, refreshed on demand.

            The report recompiles a routing and reloads configuration, so it is
            fetched on an explicit button press rather than on the 250 ms poll.
            """

            page = QWidget()
            root = QVBoxLayout(page)
            header = QHBoxLayout()
            self.flexibility_summary = QLabel("点击右侧按钮生成柔性评估")
            self.flexibility_summary.setWordWrap(True)
            self.flexibility_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
            header.addWidget(self.flexibility_summary, 1)
            refresh = QPushButton("刷新柔性评估")
            refresh.clicked.connect(self.refresh_flexibility)
            header.addWidget(refresh)
            root.addLayout(header)

            demo_group = QGroupBox("可执行柔性演示（直接驱动 V2 运行时与 MuJoCo）")
            demo_layout = QGridLayout(demo_group)
            demos = (
                ("A/B/C 混流", "product_mix", "产品柔性：三个不同规格订单进入同一实体流程"),
                ("双安装支路并行", "resource_parallel", "资源柔性：Arm1/Arm3 在线选择安装支路"),
                ("三件炉批", "batch_three", "批量柔性：三托盘逐件入炉后合批"),
                ("紧急插单", "urgent_insert", "订单柔性：当前动作不抢占，下一次派工优先紧急单"),
                ("漏涂闭环", "fault_loop", "扰动柔性：先形成缺口，再由相机检测、返工、复检"),
            )
            self.flexibility_demo_buttons = {}
            for index, (title, demo, tip) in enumerate(demos):
                button = QPushButton(title)
                button.setToolTip(tip)
                button.clicked.connect(
                    lambda _=False, value=demo: self.post(
                        "/flexibility/demo",
                        {"demo": value},
                    )
                )
                button.setEnabled(selected_profile.profile_id == "V2_DUAL_INSTALL")
                self.flexibility_demo_buttons[demo] = button
                demo_layout.addWidget(button, index // 3, index % 3)
            for index, (title, tip) in enumerate(
                (
                    ("AND/OR 工艺路线", "V2 物理运行时尚未消费 OR 分支；当前只允许在订单预览中查看"),
                    ("实体自动换型", "V2 换型龙门尚未接入实体执行，禁止用瞬时显隐冒充换型"),
                    ("CP-SAT / 拍卖调度", "当前 V2 使用在线最早完成分派；优化器接口完成后再开放"),
                )
            ):
                button = QPushButton(title)
                button.setEnabled(False)
                button.setToolTip(tip)
                demo_layout.addWidget(button, 2, index)
            root.addWidget(demo_group)

            self.flexibility_table = QTableWidget(0, 4)
            self.flexibility_table.setHorizontalHeaderLabels(["柔性维度", "状态", "关键指标", "依据"])
            self.flexibility_table.horizontalHeader().setStretchLastSection(True)
            root.addWidget(self.flexibility_table, 1)

            self.flexibility_detail = QTableWidget(0, 5)
            self.flexibility_detail.setHorizontalHeaderLabels(["维度", "对象", "候选/分支", "数值", "说明"])
            self.flexibility_detail.horizontalHeader().setStretchLastSection(True)
            root.addWidget(self.flexibility_detail, 1)
            self.tabs.addTab(self._scroll_page(page), "柔性总览")

        def refresh_flexibility(self) -> None:
            try:
                report = get_json(base_url + "/flexibility", timeout=5.0)
            except Exception as exc:  # noqa: BLE001 - surfaced to the operator
                self.flexibility_summary.setText(f"柔性评估获取失败：{exc}")
                return
            summary = report.get("summary", {})
            self.flexibility_summary.setText(
                f"产线剖面 {report.get('line_profile', '-')} ｜ "
                f"共 {summary.get('total', 0)} 个柔性维度："
                f"已实现 {summary.get('full', 0)} ｜ "
                f"部分实现 {summary.get('partial', 0)} ｜ "
                f"未实现 {summary.get('none', 0)}"
            )
            dimensions = [item for item in report.get("dimensions", []) if isinstance(item, dict)]
            self._fill_table(
                self.flexibility_table,
                [
                    [
                        item.get("label_zh", "-"),
                        item.get("state_zh", "-"),
                        item.get("headline_zh", "-"),
                        item.get("evidence_zh", "-"),
                    ]
                    for item in dimensions
                ],
            )
            self._fill_table(self.flexibility_detail, self._flexibility_detail_rows(dimensions))

        @staticmethod
        def _flexibility_detail_rows(dimensions: list[dict[str, Any]]) -> list[list[Any]]:
            rows: list[list[Any]] = []
            for item in dimensions:
                label = item.get("label_zh", "-")
                metrics = item.get("metrics", {}) or {}
                for product in metrics.get("products", []):
                    rows.append(
                        [
                            label,
                            f"{product.get('preset')} 型",
                            f"{product.get('comb_module')}",
                            f"{product.get('fin_count')} 片 / {product.get('path_count')} 条",
                            f"节距 {product.get('fin_pitch_mm')} mm，"
                            f"压紧 {product.get('clamping_force_n')} N",
                        ]
                    )
                for route in metrics.get("routes", []):
                    for branch in route.get("branches", []):
                        rows.append(
                            [
                                label,
                                route.get("operation_id", "-"),
                                branch.get("mode", "-"),
                                f"{float(branch.get('duration_s', 0.0)):.1f}s",
                                (
                                    "可用：" + ", ".join(branch.get("resources", []))
                                    if branch.get("available")
                                    else "不可用：" + ("；".join(branch.get("reasons", [])) or "无可用资源")
                                ),
                            ]
                        )
                for binding in metrics.get("bindings", []):
                    if len(binding.get("candidates", [])) <= 1 and not binding.get("rejected"):
                        continue
                    rows.append(
                        [
                            label,
                            binding.get("operation_id", "-"),
                            ", ".join(
                                f"{candidate.get('resource_id')}"
                                f"({float(candidate.get('duration', 0.0)):.1f}s)"
                                for candidate in binding.get("candidates", [])
                            )
                            or "无候选",
                            f"{len(binding.get('candidates', []))} 个候选",
                            "；".join(str(entry.get("reason", "")) for entry in binding.get("rejected", []))
                            or "-",
                        ]
                    )
                for tier in metrics.get("tiers", []):
                    rows.append(
                        [
                            label,
                            tier.get("label_zh", "-"),
                            f"{tier.get('changeover_count', 0)} 次换型",
                            f"{float(tier.get('changeover_seconds', 0.0)):.1f}s",
                            (
                                "基线为占位值，需以现场实测替换"
                                if tier.get("name") == "MANUAL_TEACHING"
                                else "-"
                            ),
                        ]
                    )
            return rows

        def _build_planning_tabs(self) -> None:
            self._build_flexibility_tab()
            order_page = QWidget()
            order_root = QVBoxLayout(order_page)
            form_group = QGroupBox("运行时订单规划（不会改写YAML）")
            form = QFormLayout(form_group)
            self.order_id_input = QLineEdit("UI_ORDER_001")
            self.order_mode_input = QComboBox()
            self.order_mode_input.addItem("预设产品", "preset")
            self.order_mode_input.addItem("自定义产品", "custom")
            if selected_profile.profile_id == "V2_DUAL_INSTALL":
                custom_item = self.order_mode_input.model().item(1)
                custom_item.setEnabled(False)
                custom_item.setToolTip("V2 当前只执行已经过实体工装验证的 A/B/C 产品")
            self.preset_input = QComboBox()
            self.preset_input.addItems(["A", "B", "C"])
            self.quantity_input = QSpinBox()
            self.quantity_input.setRange(1, 3)
            self.priority_input = QSpinBox()
            self.priority_input.setRange(0, 100)
            self.priority_input.setValue(10)
            self.due_input = QLineEdit()
            self.due_input.setPlaceholderText("ISO-8601，可留空")
            self.layer_input = QComboBox()
            self.layer_input.addItems(["自动", "第1层", "第2层", "第3层"])
            self.route_strategy_input = QComboBox()
            self.route_strategy_input.addItem("标准路线", "STANDARD")
            self.route_strategy_input.addItem("高可靠路线", "HIGH_RELIABILITY")
            self.route_strategy_input.addItem("首件高可靠", "FIRST_ARTICLE")
            if selected_profile.profile_id == "V2_DUAL_INSTALL":
                self.layer_input.setEnabled(False)
                self.layer_input.setToolTip("V2 由炉前实时空层状态分配，不接受静态层位占用")
                for index in (1, 2):
                    item = self.route_strategy_input.model().item(index)
                    item.setEnabled(False)
                    item.setToolTip("该 AND/OR 路线尚未接入 V2 物理运行时，只可在柔性报告中评估")
            form.addRow("订单ID", self.order_id_input)
            form.addRow("规划模式", self.order_mode_input)
            form.addRow("产品", self.preset_input)
            form.addRow("数量", self.quantity_input)
            form.addRow("优先级", self.priority_input)
            form.addRow("交期", self.due_input)
            form.addRow("首选料架层", self.layer_input)
            form.addRow("工艺路线", self.route_strategy_input)
            order_root.addWidget(form_group)

            self.custom_product_group = QGroupBox("自定义产品与工艺参数（仅匹配实体模块后才允许执行）")
            custom_form = QFormLayout(self.custom_product_group)

            def millimetres(value: float, minimum: float, maximum: float) -> Any:
                control = QDoubleSpinBox()
                control.setRange(minimum, maximum)
                control.setDecimals(2)
                control.setValue(value)
                control.setSuffix(" mm")
                return control

            self.custom_base_l = millimetres(360.0, 100.0, 440.0)
            self.custom_base_w = millimetres(220.0, 80.0, 290.0)
            self.custom_base_t = millimetres(8.0, 1.0, 20.0)
            self.custom_fin_l = millimetres(300.0, 80.0, 380.0)
            self.custom_fin_t = millimetres(2.0, 0.5, 8.0)
            self.custom_fin_h = millimetres(60.0, 10.0, 110.0)
            self.custom_fin_count = QSpinBox()
            self.custom_fin_count.setRange(1, 12)
            self.custom_fin_count.setValue(5)
            self.custom_pitch = QComboBox()
            for value in (15, 20, 30, 40):
                self.custom_pitch.addItem(f"{value} mm", value)
            self.custom_pitch.setCurrentIndex(1)
            self.custom_margin = millimetres(15.0, 2.0, 60.0)
            self.custom_path_width = millimetres(4.0, 1.0, 10.0)
            self.custom_nozzle_spacing = millimetres(5.0, 2.0, 12.0)
            self.custom_nozzle_height = millimetres(4.0, 1.0, 20.0)
            self.custom_material_speed = QDoubleSpinBox()
            self.custom_material_speed.setRange(0.005, 0.250)
            self.custom_material_speed.setDecimals(3)
            self.custom_material_speed.setValue(0.040)
            self.custom_material_speed.setSuffix(" m/s")
            self.custom_clamp_force = QDoubleSpinBox()
            self.custom_clamp_force.setRange(5.0, 60.0)
            self.custom_clamp_force.setValue(20.0)
            self.custom_clamp_force.setSuffix(" N")
            custom_form.addRow("基板长", self.custom_base_l)
            custom_form.addRow("基板宽", self.custom_base_w)
            custom_form.addRow("基板厚", self.custom_base_t)
            custom_form.addRow("翅片长", self.custom_fin_l)
            custom_form.addRow("翅片厚", self.custom_fin_t)
            custom_form.addRow("翅片高", self.custom_fin_h)
            custom_form.addRow("翅片数量", self.custom_fin_count)
            custom_form.addRow("实体梳齿节距", self.custom_pitch)
            custom_form.addRow("路径边距", self.custom_margin)
            custom_form.addRow("焊道宽度", self.custom_path_width)
            custom_form.addRow("双喷嘴中心距", self.custom_nozzle_spacing)
            custom_form.addRow("喷嘴高度", self.custom_nozzle_height)
            custom_form.addRow("涂覆速度", self.custom_material_speed)
            custom_form.addRow("目标压紧力", self.custom_clamp_force)
            self.custom_product_group.setVisible(False)
            self.order_mode_input.currentIndexChanged.connect(
                lambda: self.custom_product_group.setVisible(self.order_mode_input.currentData() == "custom")
            )
            order_root.addWidget(self.custom_product_group)
            actions = QHBoxLayout()
            preview = QPushButton("校验并预览")
            preview.clicked.connect(self.preview_order)
            normal = QPushButton("加入普通订单")
            normal.clicked.connect(lambda: self.insert_order(False))
            urgent = QPushButton("插入紧急订单")
            urgent.clicked.connect(lambda: self.insert_order(True))
            actions.addWidget(preview)
            actions.addWidget(normal)
            actions.addWidget(urgent)
            order_root.addLayout(actions)
            self.order_action_status = QLabel("尚未提交运行时订单")
            self.order_action_status.setWordWrap(True)
            self.order_action_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
            order_root.addWidget(self.order_action_status)
            self.plan_summary = QLabel("尚未生成计划")
            self.plan_summary.setWordWrap(True)
            self.plan_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
            order_root.addWidget(self.plan_summary)
            self.order_table = QTableWidget(0, 7)
            self.order_table.setHorizontalHeaderLabels(
                ["订单", "产品", "数量", "优先级", "状态", "进度", "紧急"]
            )
            order_root.addWidget(self.order_table, 1)
            self.tabs.addTab(self._scroll_page(order_page), "订单规划")

            task_page = QWidget()
            self.task_page = task_page
            task_root = QVBoxLayout(task_page)
            self.scheduler_summary = QLabel("scheduler: FIXED_SEQUENCE")
            self.scheduler_summary.setWordWrap(True)
            task_root.addWidget(self.scheduler_summary)
            task_filters = QHBoxLayout()
            task_filters.addWidget(QLabel("工位筛选"))
            self.task_station_filter = QComboBox()
            self.task_station_filter.addItem("全部工位", "")
            for station_id, title in selected_profile.station_titles.items():
                self.task_station_filter.addItem(title, station_id)
            task_filters.addWidget(self.task_station_filter)
            task_filters.addWidget(QLabel("托盘筛选"))
            self.task_tray_filter = QComboBox()
            self.task_tray_filter.addItem("全部托盘", "")
            tray_count = 6 if selected_profile.profile_id == "V2_DUAL_INSTALL" else 4
            tray_prefix = "V2_TRAY" if selected_profile.profile_id == "V2_DUAL_INSTALL" else "tray"
            for index in range(1, tray_count + 1):
                self.task_tray_filter.addItem(
                    f"托盘{index}",
                    f"{tray_prefix}_{index:02d}",
                )
            task_filters.addWidget(self.task_tray_filter)
            task_filters.addStretch(1)
            self.task_station_filter.currentIndexChanged.connect(self._refresh_task_graph)
            self.task_tray_filter.currentIndexChanged.connect(self._refresh_task_graph)
            task_root.addLayout(task_filters)
            self.task_graph = TaskGraphView()
            task_root.addWidget(self.task_graph, 1)
            self.scheduler_decisions = QTableWidget(0, 5)
            self.scheduler_decisions.setHorizontalHeaderLabels(
                ["任务", "候选资源", "总成本", "是否选中", "阻塞原因"]
            )
            task_root.addWidget(self.scheduler_decisions)
            self.tabs.addTab(self._scroll_page(task_page), "任务图 / 调度")

            pipeline_page = QWidget()
            pipeline_root = QVBoxLayout(pipeline_page)
            pipeline_title = (
                "V2 双安装支路异步流水工位"
                if selected_profile.profile_id == "V2_DUAL_INSTALL"
                else "浅U型异步流水工位"
            )
            pipeline_overview = QGroupBox(pipeline_title)
            pipeline_grid = QGridLayout(pipeline_overview)
            self.async_station_labels = {}
            for index, (station_id, title) in enumerate(selected_profile.station_titles.items()):
                label = QLabel(f"{title}：空闲")
                label.setWordWrap(True)
                self.async_station_labels[station_id] = label
                pipeline_grid.addWidget(label, index // 2, index % 2)
            initial_route = (
                "单向流：S1 → S2A → S2B → S3A/S3B → Y合流 → S4 → 三层炉 | WIP 0/6"
                if selected_profile.profile_id == "V2_DUAL_INSTALL"
                else "单向流：S1 → S2A → S2B → S3 → 料架 | WIP 0/3"
            )
            self.async_line_status = QLabel(initial_route)
            self.async_line_status.setWordWrap(True)
            pipeline_grid.addWidget(
                self.async_line_status,
                (len(selected_profile.station_titles) + 1) // 2,
                0,
                1,
                2,
            )
            pipeline_root.addWidget(pipeline_overview)
            self.transfer_table = QTableWidget(0, 7)
            self.transfer_table.setHorizontalHeaderLabels(
                ["移载段", "起点", "终点", "托盘", "状态", "进度", "物理位置"]
            )
            pipeline_root.addWidget(self.transfer_table)
            self.tray_route_table = QTableWidget(0, 8)
            self.tray_route_table.setHorizontalHeaderLabels(
                ["托盘", "订单/工件", "唯一归属", "当前工位", "阶段", "在制", "模具/梳齿", "压紧"]
            )
            pipeline_root.addWidget(self.tray_route_table)
            self.motion_table = QTableWidget(0, 7)
            self.motion_table.setHorizontalHeaderLabels(
                ["机械臂", "请求", "规划器", "起始", "结束", "等待", "预约"]
            )
            pipeline_root.addWidget(self.motion_table)
            self.tabs.addTab(self._scroll_page(pipeline_page), "异步流水工位")

            gantt_page = QWidget()
            gantt_root = QVBoxLayout(gantt_page)
            gantt_note = QLabel(
                "实时甘特数据来自任务的READY/RUNNING/SUCCEEDED时间戳；等待与路径冲突不使订单直接ERROR。"
            )
            gantt_note.setWordWrap(True)
            gantt_root.addWidget(gantt_note)
            self.gantt_table = QTableWidget(0, 10)
            self.gantt_table.setHorizontalHeaderLabels(
                [
                    "资源",
                    "任务",
                    "工位",
                    "托盘",
                    "状态",
                    "计划时长",
                    "实际开始",
                    "实际结束",
                    "等待",
                    "冲突/说明",
                ]
            )
            gantt_root.addWidget(self.gantt_table, 1)
            self.tabs.addTab(self._scroll_page(gantt_page), "实时甘特图")

            drawing_page = QWidget()
            drawing_root = QVBoxLayout(drawing_page)
            drawing_note = QLabel("由当前ProcessPlan生成俯视几何、焊缝和关键尺寸示意。")
            drawing_root.addWidget(drawing_note)
            self.engineering_drawing = EngineeringDrawing()
            drawing_root.addWidget(self.engineering_drawing, 1)
            drawing_actions = QHBoxLayout()
            png = QPushButton("导出PNG")
            png.clicked.connect(lambda: self.export_drawing("png"))
            svg = QPushButton("导出SVG")
            svg.clicked.connect(lambda: self.export_drawing("svg"))
            drawing_actions.addWidget(png)
            drawing_actions.addWidget(svg)
            drawing_root.addLayout(drawing_actions)
            self.tabs.addTab(self._scroll_page(drawing_page), "产品工程图规划")

            resource_page = QWidget()
            resource_root = QVBoxLayout(resource_page)
            self.resource_table = QTableWidget(0, 7)
            self.resource_table.setHorizontalHeaderLabels(
                ["资源", "类型", "状态", "任务", "工具", "故障", "区域"]
            )
            resource_root.addWidget(self.resource_table, 1)
            self.zone_status = QLabel("区域锁：-")
            self.zone_status.setWordWrap(True)
            resource_root.addWidget(self.zone_status)
            self.tabs.addTab(self._scroll_page(resource_page), "资源与区域")

            recovery_page = QWidget()
            recovery_root = QVBoxLayout(recovery_page)
            injection_group = QGroupBox("手动故障注入台")
            injection_root = QVBoxLayout(injection_group)
            injection_form = QFormLayout()
            self.fault_type_input = QComboBox()
            for definition in MANUAL_FAULT_CATALOG.values():
                self.fault_type_input.addItem(
                    f"[{definition.category_zh}] {definition.label_zh}",
                    definition.fault_type,
                )
            self.fault_type_input.currentIndexChanged.connect(self._fault_type_changed)
            self.fault_target_input = QComboBox()
            self.fault_severity_input = QComboBox()
            self.fault_severity_input.addItem("可恢复（自动生成修复流程）", "recoverable")
            self.fault_severity_input.addItem("严重（人工复核/报废）", "severe")
            self.fault_severity_input.currentIndexChanged.connect(self._fault_severity_changed)
            self.fault_auto_recover = QCheckBox("设备恢复后自动继续当前物理流程")
            self.fault_auto_recover.setChecked(True)
            self.fault_duration_input = QSpinBox()
            self.fault_duration_input.setRange(1, 600)
            self.fault_duration_input.setValue(8)
            self.fault_duration_input.setSuffix(" 仿真秒")
            injection_form.addRow("故障类型", self.fault_type_input)
            injection_form.addRow("故障目标", self.fault_target_input)
            injection_form.addRow("严重度", self.fault_severity_input)
            injection_form.addRow("自动恢复", self.fault_auto_recover)
            injection_form.addRow("离线/停顿时间", self.fault_duration_input)
            injection_root.addLayout(injection_form)
            self.fault_hint = QLabel()
            self.fault_hint.setWordWrap(True)
            self.fault_hint.setStyleSheet(
                "padding:8px;background:#16202a;border:1px solid #334155;color:#d8e6f3"
            )
            injection_root.addWidget(self.fault_hint)
            injection_actions = QHBoxLayout()
            start_demo = QPushButton("先启动A型故障演示")
            start_demo.clicked.connect(lambda: self.post("/order", {"preset": "A"}))
            injection_actions.addWidget(start_demo)
            inject = QPushButton("注入所选故障")
            inject.setStyleSheet("font-weight:bold;padding:7px;background:#a33a2b;color:white")
            inject.clicked.connect(self.inject_selected_fault)
            injection_actions.addWidget(inject)
            self.recover_selected_arm = QPushButton("立即恢复所选机械臂")
            self.recover_selected_arm.clicked.connect(self.recover_selected_arm_fault)
            injection_actions.addWidget(self.recover_selected_arm)
            injection_actions.addStretch(1)
            injection_root.addLayout(injection_actions)
            quick_row = QHBoxLayout()
            quick_actions = (
                ("快速：翅片偏位", "FIN_POSE", None, "recoverable"),
                ("快速：漏涂", "BRAZING_MISSING", None, "recoverable"),
                ("快速：Arm2离线", "ARM_UNAVAILABLE", "ARM2", "recoverable"),
                ("快速：二层不可用", "RACK_LAYER_UNAVAILABLE", "1", "recoverable"),
                ("快速：严重炉温", "FURNACE_PROFILE", "furnace", "severe"),
            )
            for title, fault_type, target, severity in quick_actions:
                button = QPushButton(title)
                button.clicked.connect(
                    lambda _=False, kind=fault_type, value=target, level=severity: self.quick_fault(
                        kind, value, level
                    )
                )
                quick_row.addWidget(button)
            quick_row.addStretch(1)
            injection_root.addLayout(quick_row)
            self.fault_injection_result = QLabel("选择故障后点击注入；系统会等待正确工序再触发。")
            self.fault_injection_result.setWordWrap(True)
            injection_root.addWidget(self.fault_injection_result)
            recovery_root.addWidget(injection_group)
            if selected_profile.profile_id == "V2_DUAL_INSTALL":
                injection_group.setToolTip("V2 完整闭环：工序形成物理缺陷 → 相机检测 → 托盘返回返工 → 再检测")

            self.manual_fault_table = QTableWidget(0, 6)
            self.manual_fault_table.setHorizontalHeaderLabels(
                ["注入请求", "故障", "目标", "状态", "触发时间", "自动恢复"]
            )
            recovery_root.addWidget(self.manual_fault_table)
            self.physical_fault_table = QTableWidget(0, 8)
            self.physical_fault_table.setHorizontalHeaderLabels(
                ["物理缺陷", "类型", "目标", "形成工序", "检测工序", "阶段", "形成时间", "检测/修复"]
            )
            self.physical_fault_table.horizontalHeader().setStretchLastSection(True)
            recovery_root.addWidget(self.physical_fault_table)
            self.fault_table = QTableWidget(0, 6)
            self.fault_table.setHorizontalHeaderLabels(
                ["故障ID", "类型", "来源", "目标/任务", "可恢复", "状态"]
            )
            recovery_root.addWidget(self.fault_table)
            self.recovery_table = QTableWidget(0, 6)
            self.recovery_table.setHorizontalHeaderLabels(["恢复ID", "策略", "状态", "重试", "步骤", "信息"])
            self.recovery_table.itemSelectionChanged.connect(self._select_recovery)
            recovery_root.addWidget(self.recovery_table)
            recovery_actions = QHBoxLayout()
            for title, action in (
                ("暂停恢复", "pause"),
                ("继续恢复", "resume"),
                ("重新尝试", "retry"),
                ("转人工", "manual_review"),
            ):
                button = QPushButton(title)
                button.clicked.connect(lambda _=False, value=action: self.recovery_action(value))
                recovery_actions.addWidget(button)
            replan = QPushButton("手动重规划")
            replan.clicked.connect(lambda: self.post("/scheduler/replan", {"reason": "qt_operator"}))
            # V2 has no replanner: its dispatcher re-evaluates every tick, so a
            # manual replan request has nothing to trigger.
            if selected_profile.profile_id == "V2_DUAL_INSTALL":
                replan.setEnabled(False)
                replan.setToolTip("V2 调度器每 tick 自动重算，无需手动重规划")
            recovery_actions.addWidget(replan)
            recovery_root.addLayout(recovery_actions)
            self.tabs.addTab(self._scroll_page(recovery_page), "故障与恢复规划")
            self._fault_type_changed()

            logistics_page = QWidget()
            logistics_root = QVBoxLayout(logistics_page)
            self.logistics_batch = QLabel("批次：-")
            self.logistics_rack = QLabel("料架：EMPTY | EMPTY | EMPTY")
            self.logistics_transfer = QLabel("移载：IDLE")
            self.logistics_furnace = QLabel("炉体：IDLE")
            for label in (
                self.logistics_batch,
                self.logistics_rack,
                self.logistics_transfer,
                self.logistics_furnace,
            ):
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                logistics_root.addWidget(label)
            logistics_root.addStretch(1)
            self.tabs.addTab(self._scroll_page(logistics_page), "批次与物流")

            metrics_page = QWidget()
            metrics_root = QVBoxLayout(metrics_page)
            self.metrics_text = QLabel("尚无V2实验指标")
            self.metrics_text.setWordWrap(True)
            self.metrics_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
            metrics_root.addWidget(self.metrics_text)
            self.metrics_table = QTableWidget(0, 4)
            self.metrics_table.setHorizontalHeaderLabels(
                (
                    ["指标", "基线", "当前 V2", "数据来源"]
                    if selected_profile.profile_id == "V2_DUAL_INSTALL"
                    else ["指标", "Fixed", "Dynamic", "变化"]
                )
            )
            metrics_root.addWidget(self.metrics_table, 1)
            self.tabs.addTab(self._scroll_page(metrics_page), "指标与实验")

        def _order_payload(self, urgent: bool = False) -> dict[str, Any]:
            layer_index = self.layer_input.currentIndex() - 1
            due = self.due_input.text().strip()
            payload = {
                "order_id": self.order_id_input.text().strip(),
                "mode": str(self.order_mode_input.currentData()),
                "preset": self.preset_input.currentText(),
                "quantity": self.quantity_input.value(),
                "priority": self.priority_input.value(),
                "due_time": due or None,
                "preferred_rack_layer": None if layer_index < 0 else layer_index,
                "urgent": urgent,
                "route_strategy": str(self.route_strategy_input.currentData()),
            }
            if self.order_mode_input.currentData() == "custom":
                millimetre = 0.001
                payload["custom_product"] = {
                    "base_size_m": [
                        millimetre * self.custom_base_l.value(),
                        millimetre * self.custom_base_w.value(),
                        millimetre * self.custom_base_t.value(),
                    ],
                    "fin_size_m": [
                        millimetre * self.custom_fin_l.value(),
                        millimetre * self.custom_fin_t.value(),
                        millimetre * self.custom_fin_h.value(),
                    ],
                    "fin_count": self.custom_fin_count.value(),
                    "fin_pitch_m": millimetre * float(self.custom_pitch.currentData()),
                    "path_margin_m": millimetre * self.custom_margin.value(),
                    "path_width_m": millimetre * self.custom_path_width.value(),
                    "nozzle_spacing_m": millimetre * self.custom_nozzle_spacing.value(),
                    "nozzle_tip_height_m": millimetre * self.custom_nozzle_height.value(),
                    "material_speed_m_s": self.custom_material_speed.value(),
                    "target_clamping_force_n": self.custom_clamp_force.value(),
                    "recipe": "demo_brazing",
                }
            return payload

        def preview_order(self) -> None:
            try:
                response = post_json(base_url + "/orders/plan", self._order_payload(), timeout=1.0)
                self.current_plan = dict(response.get("plan", {}))
                self.engineering_drawing.set_plan(self.current_plan)
                self.task_graph.set_tasks(list(response.get("task_preview", [])))
                self.plan_summary.setText(
                    f"{self.current_plan.get('order_id')} | {self.current_plan.get('product_id')} | "
                    f"{self.current_plan.get('quantity')}件 | 翅片{self.current_plan.get('fin_count')} | "
                    f"路径{self.current_plan.get('path_count')} | 任务{self.current_plan.get('estimated_task_count')} | "
                    f"层位{[int(v)+1 for v in self.current_plan.get('rack_layers', [])]}"
                )
            except Exception as exc:
                self.plan_summary.setText(f"计划校验失败：{exc}")

        def insert_order(self, urgent: bool) -> None:
            payload = self._order_payload(urgent)
            known_ids = set(self._submitted_order_ids)
            for item in self.latest_state.get("orders", []):
                if isinstance(item, dict) and item.get("order_id"):
                    known_ids.add(str(item["order_id"]))
            payload["order_id"] = unique_order_id(str(payload.get("order_id", "")), known_ids)
            kind = "紧急订单" if urgent else "普通订单"
            try:
                response = post_json(base_url + "/orders/insert", payload, timeout=0.8)
                submitted_id = str(response.get("order_id", payload["order_id"]))
                self._submitted_order_ids.add(submitted_id)
                self.order_action_status.setText(f"✓ {kind} {submitted_id} 已加入执行队列")
                self.result.setText(f"订单提交成功：{submitted_id}")
                # Prepare a fresh id immediately. This also protects rapid
                # consecutive clicks before the next /state refresh arrives.
                known_ids.update(self._submitted_order_ids)
                self.order_id_input.setText(unique_order_id(submitted_id, known_ids))
            except Exception as exc:
                message = f"✗ {kind}提交失败：{exc}"
                self.order_action_status.setText(message)
                self.result.setText(message)

        def export_drawing(self, extension: str) -> None:
            if not self.current_plan:
                self.result.setText("请先预览订单，再导出工程示意")
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "导出工程示意",
                f"{self.current_plan.get('order_id', 'plan')}.{extension}",
                f"{extension.upper()} (*.{extension})",
            )
            if not path:
                return
            try:
                if extension == "png":
                    if not self.engineering_drawing.grab().save(path, "PNG"):
                        raise RuntimeError("PNG保存失败")
                else:
                    generator = QSvgGenerator()
                    generator.setFileName(path)
                    generator.setSize(self.engineering_drawing.size())
                    generator.setViewBox(self.engineering_drawing.rect())
                    painter = QPainter(generator)
                    self.engineering_drawing.render(painter)
                    painter.end()
                self.result.setText(f"工程示意已导出：{path}")
            except Exception as exc:
                self.result.setText(f"导出失败：{exc}")

        def _select_recovery(self) -> None:
            row = self.recovery_table.currentRow()
            item = self.recovery_table.item(row, 0) if row >= 0 else None
            self.current_recovery_id = "" if item is None else item.text()

        def recovery_action(self, action: str) -> None:
            if not self.current_recovery_id:
                self.result.setText("请先在恢复计划表中选择一项")
                return
            self.post(f"/recoveries/{self.current_recovery_id}/action", {"action": action})

        def _fault_definition(self) -> Any:
            return MANUAL_FAULT_CATALOG[str(self.fault_type_input.currentData())]

        def _fault_type_changed(self, *_: Any) -> None:
            definition = self._fault_definition()
            safety_fault = definition.fault_type in {
                "CONTACT_SAFETY_STOP",
                "TRAY_STATE_INCONSISTENT",
            }
            simulated_manual_review = bool(definition.simulated_manual_review)
            severity_index = self.fault_severity_input.findData("severe" if safety_fault else "recoverable")
            if severity_index >= 0:
                self.fault_severity_input.setCurrentIndex(severity_index)
            self.fault_auto_recover.setChecked(not safety_fault and not simulated_manual_review)
            self.fault_hint.setText(
                f"触发说明：{definition.hint_zh}\n"
                + (
                    "提示：故障触发或被相机检出后，将弹出人工审核窗口并用10秒仿真时间模拟修复。"
                    if simulated_manual_review
                    else "提示：点击后先进入等待；生产工序会形成可见缺陷，相机完成检测后才启动返工与复检。"
                )
            )
            self.fault_duration_input.setEnabled(definition.supports_duration and not simulated_manual_review)
            self.fault_auto_recover.setEnabled(
                definition.runtime_fault is not None and not safety_fault and not simulated_manual_review
            )
            self.recover_selected_arm.setEnabled(definition.target_kind == "arm")
            self._fault_target_signature = ()
            self._refresh_fault_targets(self.latest_state)

        def _fault_severity_changed(self, *_: Any) -> None:
            if self.fault_severity_input.currentData() == "severe":
                self.fault_auto_recover.setChecked(False)

        def _fault_target_values(self, state: dict[str, Any]) -> list[tuple[str, str]]:
            definition = self._fault_definition()
            if definition.target_kind == "fin":
                values = [
                    (name, name)
                    for name, item in sorted(state.get("fins", {}).items())
                    if isinstance(item, dict) and item.get("active", False)
                ]
                return values or [(f"fin_{index:02d}", f"fin_{index:02d}") for index in range(1, 6)]
            if definition.target_kind == "path":
                values = [
                    (name, name)
                    for name, item in sorted(state.get("paths", {}).items())
                    if isinstance(item, dict) and item.get("active", False)
                ]
                return values or [("slot_01_left", "slot_01_left")]
            if definition.target_kind == "arm":
                return [("Arm1", "ARM1"), ("Arm2", "ARM2"), ("Arm3", "ARM3")]
            if definition.target_kind == "rack_layer":
                return [("第1层", "0"), ("第2层", "1"), ("第3层", "2")]
            if definition.target_kind == "furnace_conveyor":
                return [("炉前黑色传送带", "FURNACE_CONVEYOR")]
            if definition.target_kind == "furnace":
                return [("钎焊炉/炉门", "furnace")]
            active = next(
                (
                    str(task.get("display_name_zh") or task.get("task_id"))
                    for task in state.get("tasks", [])
                    if isinstance(task, dict) and task.get("status") == "RUNNING"
                ),
                "当前运行任务（系统自动匹配）",
            )
            return [(active, "")]

        def _refresh_fault_targets(self, state: dict[str, Any]) -> None:
            values = self._fault_target_values(state)
            signature = tuple(f"{label}|{value}" for label, value in values)
            if signature == self._fault_target_signature:
                return
            current = self.fault_target_input.currentData()
            self.fault_target_input.clear()
            for label, value in values:
                self.fault_target_input.addItem(label, value)
            index = self.fault_target_input.findData(current)
            if index >= 0:
                self.fault_target_input.setCurrentIndex(index)
            self._fault_target_signature = signature

        def _manual_fault_payload(self) -> dict[str, Any]:
            return {
                "fault_type": str(self.fault_type_input.currentData()),
                "target": str(self.fault_target_input.currentData() or ""),
                "severity": str(self.fault_severity_input.currentData()),
                "auto_recover": self.fault_auto_recover.isChecked(),
                "duration_s": self.fault_duration_input.value(),
            }

        def inject_selected_fault(self) -> None:
            try:
                stage = str(self.latest_state.get("stage", "IDLE"))
                if not self.latest_state.get("order_id") or stage in {
                    "IDLE",
                    "PASS",
                    "REWORK_REQUIRED",
                    "SCRAPPED",
                    "COMPLETE",
                    "ERROR",
                }:
                    raise RuntimeError("请先启动一个未结束的订单或批次，再注入故障")
                payload = self._manual_fault_payload()
                response = post_json(base_url + "/faults/inject", payload, timeout=0.8)
                definition = self._fault_definition()
                self.fault_injection_result.setText(
                    f"✓ 已受理：{definition.label_zh}；目标："
                    f"{self.fault_target_input.currentText() or '自动匹配'}。"
                    "等待正确工序触发，进度可在下方注入请求和任务图中查看。"
                )
                self.result.setText(f"故障注入已受理：{response.get('label_zh', definition.label_zh)}")
            except Exception as exc:
                self.fault_injection_result.setText(f"✗ 注入失败：{exc}")

        def quick_fault(self, fault_type: str, target: str | None, severity: str) -> None:
            index = self.fault_type_input.findData(fault_type)
            if index < 0:
                return
            self.fault_type_input.setCurrentIndex(index)
            self._refresh_fault_targets(self.latest_state)
            if target is not None:
                target_index = self.fault_target_input.findData(target)
                if target_index >= 0:
                    self.fault_target_input.setCurrentIndex(target_index)
            severity_index = self.fault_severity_input.findData(severity)
            if severity_index >= 0:
                self.fault_severity_input.setCurrentIndex(severity_index)
            self.inject_selected_fault()

        def recover_selected_arm_fault(self) -> None:
            if self._fault_definition().target_kind != "arm":
                self.fault_injection_result.setText("当前故障不是机械臂离线故障")
                return
            resource = str(self.fault_target_input.currentData() or "")
            if not resource:
                self.fault_injection_result.setText("请先选择需要恢复的机械臂")
                return
            try:
                post_json(base_url + f"/resources/{resource}/recover", {}, timeout=0.8)
                self.fault_injection_result.setText(f"✓ 已请求恢复 {resource}，系统将重新规划待执行任务。")
            except Exception as exc:
                self.fault_injection_result.setText(f"✗ 资源恢复失败：{exc}")

        def _refresh_task_graph(self, *_: Any) -> None:
            tasks = self.latest_state.get("tasks", [])
            if not isinstance(tasks, list):
                return
            station = str(self.task_station_filter.currentData() or "")
            tray = str(self.task_tray_filter.currentData() or "")
            filtered = [
                task
                for task in tasks
                if isinstance(task, dict)
                and (not station or str(task.get("station_id") or "") == station)
                and (not tray or str(task.get("tray_id") or "") == tray)
            ]
            self.task_graph.set_tasks(filtered)

        def _fill_table(self, table: Any, rows: list[list[Any]]) -> None:
            self._configure_table(table)
            signature = tuple(tuple(str(value) for value in row) for row in rows)
            key = id(table)
            if self._table_signatures.get(key) == signature:
                return
            self._table_signatures[key] = signature
            table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column, value in enumerate(row):
                    table.setItem(row_index, column, QTableWidgetItem(str(value)))

        def _button(self, layout: Any, text: str, path: str, payload: dict[str, Any]) -> Any:
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, p=path, body=payload: self.post(p, body))
            layout.addWidget(button)
            return button

        def post(self, path: str, payload: dict[str, Any]) -> None:
            try:
                response = post_json(base_url + path, payload, timeout=0.5)
                command = str(response.get("segment") or response.get("type") or path)
                self.result.setText(f"command accepted: {command}")
            except Exception as exc:
                self.result.setText(f"request failed: {exc}")

        def _sync_manual_review_popup(self, state: dict[str, Any]) -> None:
            popup = manual_review_popup_state(state)
            if popup is None:
                if self._manual_review_dialog is not None:
                    self._manual_review_dialog.close()
                    self._manual_review_dialog = None
                    self._manual_review_recovery_id = ""
                return
            recovery_id = popup["recovery_id"]
            succeeded = popup["status"] == "SUCCEEDED"
            if succeeded and recovery_id in self._manual_review_success_seen:
                return
            if self._manual_review_dialog is None or self._manual_review_recovery_id != recovery_id:
                if self._manual_review_dialog is not None:
                    self._manual_review_dialog.close()
                self._manual_review_dialog = QMessageBox(self)
                self._manual_review_dialog.setWindowTitle("人工审核")
                self._manual_review_dialog.setWindowModality(Qt.WindowModal)
                self._manual_review_recovery_id = recovery_id
                self._manual_review_dialog.show()
            self._manual_review_dialog.setText(popup["message"])
            if succeeded:
                self._manual_review_dialog.setIcon(QMessageBox.Information)
                self._manual_review_dialog.setStandardButtons(QMessageBox.Ok)
                self._manual_review_success_seen.add(recovery_id)
            else:
                self._manual_review_dialog.setIcon(QMessageBox.Warning)
                self._manual_review_dialog.setStandardButtons(QMessageBox.NoButton)

        def refresh(self) -> None:
            try:
                state = get_json(base_url + "/state", timeout=0.5)
                self.latest_state = state
                self._sync_manual_review_popup(state)
                capabilities = state.get("ui_capabilities", {})
                segment_capabilities = (
                    capabilities.get("segments", {}) if isinstance(capabilities, dict) else {}
                )
                for segment, button in self.segment_buttons.items():
                    enabled = (
                        bool(segment_capabilities.get(segment, False))
                        if selected_profile.profile_id == "V2_DUAL_INSTALL"
                        else bool(segment_capabilities.get(segment, True))
                    )
                    button.setEnabled(enabled)
                    button.setToolTip("" if enabled else "对应物理工序尚未接通，禁止形式化演示")
                flexibility_actions = (
                    capabilities.get("flexibility_actions", {}) if isinstance(capabilities, dict) else {}
                )
                for action, button in self.flexibility_demo_buttons.items():
                    enabled = bool(flexibility_actions.get(action, False))
                    button.setEnabled(enabled)
                self._refresh_fault_targets(state)
                self.order.setText(f"order: {state.get('order_id') or '-'}")
                paused = " (PAUSED)" if state.get("paused", False) else ""
                self.stage.setText(f"stage: {state.get('stage', 'IDLE')}{paused}")
                speed = float(state.get("simulation_speed", 1.0))
                actual_rtf = float(state.get("simulation_actual_rtf", 0.0))
                saturation = " | 已达计算上限" if state.get("simulation_speed_saturated") else ""
                actual_text = "采样中" if actual_rtf <= 0.0 else f"{actual_rtf:.1f}×"
                self.speed.setText(f"目标速度: {speed:g}× | 实际RTF: {actual_text}{saturation}")
                fixture = state.get("fixture", {})
                resources = state.get("resources", {})
                table2 = resources.get("table2_zone", {}) if isinstance(resources, dict) else {}
                table2_owner = table2.get("owner", "-") if isinstance(table2, dict) else "-"
                self.fixture.setText(
                    f"fixture: {fixture.get('status', fixture or '-')} | "
                    f"comb: {fixture.get('active_comb_module') or '-'} | "
                    f"press: {fixture.get('press_state', '-')} "
                    f"{float(fixture.get('clamping_force_n', 0.0)):.1f} N | "
                    f"Table2: {table2_owner or '-'}"
                )
                furnace = state.get("furnace", {})
                temperature = float(furnace.get("temperature_c", 25.0))
                self.furnace.setText(f"furnace: {furnace.get('status', '-')}  {temperature:.1f} °C")
                self.plot.add(temperature)
                error = str(state.get("last_error", ""))
                controller_status = str(state.get("status", ""))
                self.result.setText(
                    f"result: {state.get('disposition') or '-'} | "
                    f"status: {controller_status or '-'}"
                    f"{' | error: ' + error if error else ''}"
                )
                for name, label in self.arms.items():
                    arm = state.get("arms", {}).get(name, {})
                    physical_detail = ""
                    if selected_profile.profile_id == "V2_DUAL_INSTALL":
                        waypoint_count = int(arm.get("waypoint_count", 0))
                        waypoint_index = int(arm.get("waypoint_index", 0))
                        target = str(arm.get("target_zh") or "")
                        if target:
                            physical_detail = (
                                f" | {target}"
                                f"{f' [{waypoint_index}/{waypoint_count}]' if waypoint_count else ''}"
                            )
                    label.setText(
                        f"{name}: {arm.get('status', 'idle')}  "
                        f"{arm.get('task_type', '')} {arm.get('task_id', '')}"
                        f"{physical_detail}"
                    )
                batch = state.get("batch", {})
                rack = state.get("rack", {})
                transfer = state.get("transfer", {})
                if isinstance(batch, dict) and batch:
                    units = batch.get("units", [])
                    unit_text = " | ".join(
                        f"L{unit.get('layer', index + 1)}:{unit.get('phase', '-')}"
                        for index, unit in enumerate(units)
                        if isinstance(unit, dict)
                    )
                    self.batch_status.setText(
                        f"batch: {batch.get('batch_id', '-')} | "
                        f"active layer: {batch.get('active_layer', '-')} | {unit_text}"
                    )
                else:
                    self.batch_status.setText("batch: -")
                shelves = rack.get("shelves", []) if isinstance(rack, dict) else []
                shelf_text = " | ".join(
                    f"L{int(shelf.get('index', index)) + 1}:"
                    f"{shelf.get('state', 'EMPTY')}"
                    f"{'[LOCK]' if shelf.get('lock_engaged', False) else ''}"
                    for index, shelf in enumerate(shelves)
                    if isinstance(shelf, dict)
                )
                self.rack_status.setText(f"rack: {shelf_text or 'EMPTY | EMPTY | EMPTY'}")
                self.logistics_batch.setText(self.batch_status.text())
                self.logistics_rack.setText(self.rack_status.text())
                if isinstance(transfer, dict):
                    prefetch_index = transfer.get("prefetch_unit_index")
                    prefetched_index = transfer.get("prefetch_complete_index")
                    if isinstance(prefetch_index, int):
                        overlap_text = f" | concurrent prefetch: L{prefetch_index + 1}"
                    elif isinstance(prefetched_index, int):
                        overlap_text = f" | prefetched: L{prefetched_index + 1}"
                    elif transfer.get("parallel_active", False):
                        overlap_text = " | concurrent axis return"
                    else:
                        overlap_text = ""
                    mechanism_text = ""
                    if selected_profile.profile_id == "V2_DUAL_INSTALL":
                        segment_count = int(transfer.get("segment_count", 1))
                        segment_index = int(transfer.get("segment_index", 1))
                        mechanism_text = (
                            f" | 运输分段: {segment_index}/{segment_count}"
                            f" | 升降: {1000.0 * float(transfer.get('lift_height_m', 0.0)):.0f} mm"
                            f" | 前推叉: {1000.0 * float(transfer.get('pusher_position_m', 0.0)):.0f} mm"
                            f" | 后抽叉: "
                            f"{1000.0 * float(transfer.get('rear_extractor_position_m', 0.0)):.0f} mm"
                        )
                    self.transfer_status.setText(
                        f"transfer: {transfer.get('phase', 'IDLE')} | "
                        f"step: {transfer.get('step') or '-'} | "
                        f"tray: {transfer.get('unit_id') or '-'} | "
                        f"入炉传送: {100.0 * float(transfer.get('conveyor_progress', 0.0)):.0f}% | "
                        f"lock: {1000.0 * float(transfer.get('lock_position_m', 0.0)):.0f} mm"
                        f"{mechanism_text}{overlap_text}"
                    )
                    self.logistics_transfer.setText(self.transfer_status.text())
                self.logistics_furnace.setText(self.furnace.text())
                fins = state.get("fins", {})
                paths = state.get("paths", {})
                fin_done = sum(bool(item.get("inserted", False)) for item in fins.values())
                path_done = sum(bool(item.get("applied", False)) for item in paths.values())
                active_fins = sum(bool(item.get("active", False)) for item in fins.values())
                active_paths = sum(bool(item.get("active", False)) for item in paths.values())
                self.progress.setText(
                    f"fins {fin_done}/{active_fins or 5} | paths {path_done}/{active_paths or 10}"
                )
                arm2_process = state.get("arm2_process", {})
                self.arm2_tool.setText(
                    "Arm2 固定工具: 双喷嘴焊料枪 | 位置: 机械臂末端 | "
                    f"path: {arm2_process.get('current_path') or '-'} | "
                    f"applied: {arm2_process.get('completed_paths', 0)}/"
                    f"{arm2_process.get('total_paths', 0)}"
                )
                conveyor = state.get("conveyor", {})
                self.conveyor.setText(
                    f"conveyor: {conveyor.get('phase', 'IDLE')} | "
                    f"position: {1000.0 * float(conveyor.get('position_m', 0.0)):.0f}/"
                    f"{1000.0 * float(conveyor.get('travel_m', 0.0)):.0f} mm"
                )
                inspections = state.get("inspections", [])
                self.inspection.setText(f"inspection: {inspections[-1] if inspections else '-'}")
                kpi = state.get("kpi", {})
                self.kpi.setText(
                    f"KPI: elapsed={float(kpi.get('order_elapsed', 0.0)):.1f}s  "
                    f"rework={kpi.get('rework_counts', {})}  score={kpi.get('final_quality_score', '-')}"
                )
                scheduler = state.get("scheduler", {})
                self.scheduler_summary.setText(
                    f"scheduler: {scheduler.get('mode', 'FIXED_SEQUENCE')} | "
                    f"READY {scheduler.get('ready_count', 0)} | "
                    f"RUNNING {scheduler.get('running_count', 0)} | "
                    f"重规划 {scheduler.get('replan_count', 0)} | "
                    f"最大并行 {scheduler.get('max_assignments_per_tick', '-') }"
                )
                tasks = state.get("tasks", [])
                current_tab = self.tabs.currentWidget()
                if (
                    isinstance(tasks, list)
                    and isinstance(current_tab, QScrollArea)
                    and current_tab.widget() is self.task_page
                ):
                    self._refresh_task_graph()
                selected_ids = {
                    str(item.get("task_id"))
                    for item in scheduler.get("selected", [])
                    if isinstance(item, dict)
                }
                decision_rows = [
                    [
                        item.get("task_id", "-"),
                        item.get("resource_id", "-"),
                        f"{float(item.get('cost', 0.0)):.3f}",
                        "是" if str(item.get("task_id")) in selected_ids else "否",
                        "-",
                    ]
                    for item in scheduler.get("candidates", [])
                    if isinstance(item, dict)
                ]
                decision_rows.extend(
                    [
                        item.get("task_id", "-"),
                        item.get("resource_id") or "-",
                        "-",
                        "否",
                        item.get("reason", "-"),
                    ]
                    for item in scheduler.get("blocked_candidates", [])
                    if isinstance(item, dict)
                )
                self._fill_table(self.scheduler_decisions, decision_rows[:80])

                workstations = state.get("workstations", {})
                for station_id, label in self.async_station_labels.items():
                    item = workstations.get(station_id, {})
                    label.setText(
                        f"{selected_profile.station_titles[station_id]}："
                        f"{item.get('tray_id') or '空'} | "
                        f"任务 {item.get('occupied_by') or '无'} | "
                        f"允许移载 {'是' if item.get('safe_for_transfer', True) else '否'}"
                    )
                async_line = state.get("async_line", {})
                physical_owners = async_line.get("physical_tray_owners", {})
                physical_summary = ", ".join(
                    f"{tray}:{owner}" for tray, owner in sorted(physical_owners.items()) if owner != "BUFFER"
                )
                router_mode = async_line.get("process_router", {}).get("mode", "")
                mode_title = {
                    "VERIFIED_PHYSICAL_QUEUE": "分段同源真实工艺",
                    "MULTI_PALLET_RUNTIME": "多托盘并行生产",
                    "V2_DUAL_INSTALL": "V2 双安装支路",
                }.get(router_mode, "标准物理流程")
                route_text = (
                    "S1 → S2A → S2B → S3A/S3B → Y合流 → S4 → 三层贯通炉"
                    if selected_profile.profile_id == "V2_DUAL_INSTALL"
                    else "S1 → S2A → S2B → S3 → 料架"
                )
                parallelism = async_line.get("parallelism", {})
                active_arms = "/".join(parallelism.get("active_arms", [])) or "无"
                self.async_line_status.setText(
                    f"执行模式：{mode_title} | 单向流：{route_text} | "
                    f"WIP {int(async_line.get('active_wip', 0))}/"
                    f"{int(async_line.get('wip_limit', 3))} | "
                    f"当前并行臂 {active_arms}（{int(parallelism.get('current_parallel_arms', 0))}台）| "
                    f"历史峰值 {int(parallelism.get('max_parallel_arms', 0))}台 | "
                    f"多臂重叠 {float(parallelism.get('multi_arm_overlap_s', 0.0)):.1f}s | "
                    f"备用托盘 {', '.join(async_line.get('spare_trays', [])) or '无'} | "
                    f"实体托盘 {physical_summary or async_line.get('station_owner') or '缓存中/无'}"
                )
                physical_positions = async_line.get("transfer_positions_m", {})
                transfer_rows = []
                for transfer_id, item in sorted(state.get("transfers", {}).items()):
                    if not isinstance(item, dict):
                        continue
                    physical_key = str(transfer_id).removeprefix("TRANSFER_")
                    transfer_rows.append(
                        [
                            transfer_id,
                            item.get("source", "-"),
                            item.get("target", "-"),
                            item.get("tray_id") or "-",
                            item.get("status", "IDLE"),
                            f"{100.0 * float(item.get('progress', 0.0)):.0f}%",
                            f"{1000.0 * float(physical_positions.get(physical_key, 0.0)):.1f} mm",
                        ]
                    )
                self._fill_table(self.transfer_table, transfer_rows)
                phase_zh = {
                    "EMPTY_BUFFER": "空托盘缓存",
                    "CHANGEOVER": "S1备料",
                    "MOLD_READY": "模具就绪",
                    "BASE_READY": "基板就绪",
                    "MATERIAL_READY": "涂覆就绪",
                    "ASSEMBLY_READY": "组装就绪",
                    "LOCKED": "已锁紧",
                    "OUTFEED": "正在出料",
                    "FURNACE": "炉内",
                    "FINISHED_GOODS": "成品出口",
                    "RETURNING": "空托盘返回",
                }
                tray_rows = []
                for tray_id, item in sorted(state.get("tray_routes", {}).items()):
                    if not isinstance(item, dict):
                        continue
                    tray_rows.append(
                        [
                            tray_id,
                            item.get("product_unit_id") or item.get("order_id") or "-",
                            item.get("owner") or "-",
                            item.get("station_id") or "-",
                            phase_zh.get(str(item.get("phase", "")), item.get("phase", "-")),
                            "是" if item.get("order_id") else "否",
                            f"{item.get('mold_name') or '-'}/{item.get('comb_name') or '-'}",
                            "已锁" if item.get("press_locked") else "未锁",
                        ]
                    )
                self._fill_table(self.tray_route_table, tray_rows)
                motion_rows = [
                    [
                        item.get("resource_id", "-"),
                        item.get("request_id", "-"),
                        item.get("planner", "-"),
                        f"{float(item.get('start_time', 0.0)):.2f}",
                        f"{float(item.get('end_time', 0.0)):.2f}",
                        f"{float(item.get('waiting_time', 0.0)):.2f}s",
                        item.get("reservation_id") or "已释放",
                    ]
                    for item in state.get("motion_plans", [])
                    if isinstance(item, dict)
                ]
                self._fill_table(self.motion_table, motion_rows[-80:])
                gantt_rows = [
                    [
                        item.get("resource_id") or "-",
                        item.get("display_name_zh") or task_type_label_zh(item.get("task_type", "")),
                        item.get("station_id") or "-",
                        item.get("tray_id") or "-",
                        task_status_label_zh(item.get("status", "")),
                        f"{float(item.get('planned_duration', 0.0)):.2f}s",
                        "-" if item.get("actual_start") is None else f"{float(item['actual_start']):.2f}",
                        "-" if item.get("actual_end") is None else f"{float(item['actual_end']):.2f}",
                        f"{float(item.get('waiting', 0.0)):.2f}s",
                        ", ".join(item.get("blockers", [])) or "-",
                    ]
                    for item in state.get("gantt_events", [])
                    if isinstance(item, dict)
                ]
                self._fill_table(self.gantt_table, gantt_rows[-200:])
                order_rows = []
                for item in state.get("orders", []):
                    if not isinstance(item, dict):
                        continue
                    order_rows.append(
                        [
                            item.get("order_id", "-"),
                            item.get("product_id", item.get("preset", "-")),
                            item.get("quantity", "-"),
                            item.get("priority", "-"),
                            item.get("status", "-"),
                            f"{100.0 * float(item.get('progress', 0.0)):.0f}%",
                            "是" if item.get("urgent") else "否",
                        ]
                    )
                self._fill_table(self.order_table, order_rows)
                resource_rows = []
                resources_v2 = state.get("resources_v2", {})
                for resource_id, item in sorted(resources_v2.items()):
                    if not isinstance(item, dict):
                        continue
                    resource_rows.append(
                        [
                            resource_id,
                            item.get("resource_type", "-"),
                            item.get("status", "-"),
                            item.get("current_task_id") or "-",
                            item.get("current_tool") or "-",
                            item.get("fault_code") or "-",
                            ",".join(item.get("occupied_zones", [])),
                        ]
                    )
                self._fill_table(self.resource_table, resource_rows)
                zone_locks = state.get("zone_locks", {})
                locked = [
                    f"{zone}:{lease.get('task_id') or lease.get('holder') or '-'}"
                    for zone, lease in zone_locks.items()
                    if isinstance(lease, dict)
                ]
                self.zone_status.setText("区域锁：" + (" | ".join(locked) if locked else "全部空闲"))
                fault_rows = []
                for item in state.get("faults_v2", []):
                    if isinstance(item, dict):
                        definition = MANUAL_FAULT_CATALOG.get(str(item.get("fault_type", "")))
                        fault_rows.append(
                            [
                                item.get("fault_id"),
                                (definition.label_zh if definition is not None else item.get("fault_type")),
                                item.get("source"),
                                item.get("related_task_id") or "-",
                                item.get("recoverable"),
                                "已恢复" if item.get("recovered") else "处理中",
                            ]
                        )
                for index, item in enumerate(state.get("faults", []), start=1):
                    if isinstance(item, dict):
                        physical_labels = {
                            "fin_pose": "翅片位置/倾斜偏差",
                            "brazing_gap": "钎料局部漏涂",
                            "furnace_profile": "炉温曲线异常",
                        }
                        fault_rows.append(
                            [
                                f"PHYSICAL_{index:03d}",
                                physical_labels.get(str(item.get("fault_type")), item.get("fault_type")),
                                "MuJoCo物理流程",
                                item.get("target") or "-",
                                item.get("severity") != "severe",
                                "已触发" if item.get("applied") else "等待工序",
                            ]
                        )
                self._fill_table(self.fault_table, fault_rows)
                manual_rows = []
                request_status_labels = {
                    "ARMED": "等待匹配工序",
                    "MANIFESTED": "缺陷已形成（待相机检测）",
                    "DETECTED": "相机已检出（待返工）",
                    "RECOVERING": "返工/复检中",
                    "FIRED": "设备故障已触发",
                    "ACTIVE": "物理流程已暂停",
                    "RECOVERED": "物理流程已恢复",
                    "MISSED": "本订单未触发",
                }
                for item in state.get("manual_fault_requests", []):
                    if not isinstance(item, dict):
                        continue
                    manual_rows.append(
                        [
                            item.get("request_id", "-"),
                            item.get("label_zh") or item.get("fault_type", "-"),
                            item.get("target") or "自动匹配",
                            request_status_labels.get(str(item.get("status", "")), item.get("status", "-")),
                            item.get("fired_at") or item.get("started_at") or "-",
                            "是" if item.get("auto_recover", item.get("recoverable", False)) else "否",
                        ]
                    )
                self._fill_table(self.manual_fault_table, manual_rows)
                physical_status_labels = {
                    "MANIFESTED": "物理缺陷可见",
                    "DETECTED": "相机已检出",
                    "REPAIRED": "已修复，等待复检",
                    "RECOVERED": "复检通过",
                }
                physical_rows = []
                for item in state.get("physical_faults", []):
                    if not isinstance(item, dict):
                        continue
                    visual_type = str(item.get("visual_type") or item.get("fault_type") or "")
                    definition = MANUAL_FAULT_CATALOG.get(visual_type)
                    detected_at = item.get("detected_at")
                    repaired_at = item.get("repaired_at")
                    physical_rows.append(
                        [
                            item.get("defect_id", "-"),
                            definition.label_zh if definition is not None else visual_type,
                            item.get("target") or "整机",
                            item.get("source_operation") or "-",
                            item.get("detection_operation") or "-",
                            physical_status_labels.get(str(item.get("status", "")), item.get("status", "-")),
                            f"{float(item.get('manifested_at', 0.0)):.2f}s",
                            (
                                f"检出 {float(detected_at):.2f}s"
                                if detected_at is not None and repaired_at is None
                                else f"修复 {float(repaired_at):.2f}s" if repaired_at is not None else "-"
                            ),
                        ]
                    )
                self._fill_table(self.physical_fault_table, physical_rows)
                recovery_rows = []
                strategy_labels = {
                    "LOCAL_BRAZING_REWORK": "局部补涂并复检",
                    "FIN_REINSTALL": "翅片重装并复检",
                    "TRANSFER_SAFE_HOME_RETRY": "移载机构回零重试",
                    "FURNACE_INTERLOCK_RECHECK": "炉门互锁复检",
                    "RACK_LAYER_REALLOCATION": "料架层重新分配",
                    "RESOURCE_REALLOCATION": "资源隔离与重新调度",
                    "MANUAL_REVIEW": "人工安全复核",
                }
                recovery_status_labels = {
                    "PLANNED": "已规划",
                    "RUNNING": "恢复中",
                    "PAUSED": "已暂停",
                    "SUCCEEDED": "恢复成功",
                    "FAILED": "恢复失败",
                    "MANUAL_REVIEW": "等待人工处理",
                }
                for item in state.get("recoveries", []):
                    if isinstance(item, dict):
                        recovery_rows.append(
                            [
                                item.get("recovery_id"),
                                strategy_labels.get(str(item.get("strategy", "")), item.get("strategy")),
                                recovery_status_labels.get(str(item.get("status", "")), item.get("status")),
                                f"{item.get('retry_count', 0)}/{item.get('retry_limit', 0)}",
                                " → ".join(
                                    task_type_label_zh(step.get("description", ""))
                                    for step in item.get("steps", [])
                                    if isinstance(step, dict)
                                ),
                                item.get("message", ""),
                            ]
                        )
                self._fill_table(self.recovery_table, recovery_rows)
                experiment = state.get("experiment_metrics", {})
                if experiment:
                    self.metrics_text.setText(
                        f"Makespan {float(experiment.get('makespan', 0.0)):.2f}s | "
                        f"吞吐 {float(experiment.get('throughput_per_sim_second', 0.0)):.4f}/s | "
                        f"平均利用率 {100*float(experiment.get('average_robot_utilization', 0.0)):.1f}% | "
                        f"恢复率 {100*float(experiment.get('recovery_rate', 0.0)):.1f}%"
                    )
                    if selected_profile.profile_id == "V2_DUAL_INSTALL":
                        self._fill_table(
                            self.metrics_table,
                            [
                                [
                                    "完工时间 / 当前时长",
                                    "-",
                                    f"{float(experiment.get('makespan', 0.0)):.2f}s",
                                    "仿真事件时间戳",
                                ],
                                [
                                    "吞吐率",
                                    "-",
                                    f"{float(experiment.get('throughput_per_sim_second', 0.0)):.4f}/s",
                                    "UNIT_COMPLETED 事件",
                                ],
                                [
                                    "机械臂平均利用率",
                                    "-",
                                    f"{100 * float(experiment.get('average_robot_utilization', 0.0)):.1f}%",
                                    "OPERATION_STARTED/COMPLETED",
                                ],
                                [
                                    "故障恢复率",
                                    "-",
                                    f"{100 * float(experiment.get('recovery_rate', 0.0)):.1f}%",
                                    "FAULT_DETECTED/恢复完成",
                                ],
                                [
                                    "已完成工件",
                                    "-",
                                    str(experiment.get("completed_units", 0)),
                                    "UNIT_COMPLETED 事件",
                                ],
                            ],
                        )
            except Exception as exc:
                self.result.setText(f"controller unavailable: {exc}")

    app = QApplication.instance() or QApplication(sys.argv[:1])
    panel = ControlPanel()
    panel.setMinimumSize(720, 480)
    screen = app.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        panel.resize(
            min(1300, max(760, int(available.width() * 0.86))),
            min(850, max(560, int(available.height() * 0.80))),
        )
    else:
        panel.resize(1300, 850)
    panel.show()
    camera = CameraWindow()
    camera.setMinimumSize(320, 240)
    camera.resize(760, 620)
    camera.show()
    # Keep Python references alive for the lifetime of the event loop.
    panel._camera_window = camera  # type: ignore[attr-defined]
    return int(app.exec())


__all__ = ["run_ui_client"]
