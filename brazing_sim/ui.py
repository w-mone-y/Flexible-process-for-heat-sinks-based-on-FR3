"""Optional Qt control panel and independent Arm3 camera window."""

from __future__ import annotations

import sys
from typing import Any

from .api import get_bytes, get_json, post_json


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
        from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print("PySide6 is not installed; run with --headless or install brazing-sim[ui].", file=sys.stderr)
        return 2

    base_url = _base_url(args_or_url)

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
            self.setWindowTitle("Arm3 inspection camera")
            layout = QVBoxLayout(self)
            self.status = QLabel("camera starting")
            self.status.setAlignment(Qt.AlignCenter)
            self.image = QLabel()
            self.image.setAlignment(Qt.AlignCenter)
            self.image.setMinimumSize(640, 480)
            self.image.setStyleSheet("background:#05070a;border:2px solid #3b4654")
            layout.addWidget(self.status)
            layout.addWidget(self.image, 1)
            self.pixmap = QPixmap()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(120)

        def refresh(self) -> None:
            try:
                state = get_json(base_url + "/state", timeout=0.35)
                payload = get_bytes(base_url + "/camera.ppm", timeout=0.35)
                pixmap = QPixmap()
                if not pixmap.loadFromData(payload, "PPM"):
                    raise ValueError("invalid camera frame")
                self.pixmap = pixmap
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

    class ControlPanel(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("低压配电柜散热组件钎焊 MuJoCo 仿真")
            root = QVBoxLayout(self)

            controls = QGroupBox("单段流程控制")
            row = QHBoxLayout(controls)
            self._button(row, "单独运行取放", "/segment", {"segment": "pick_place"})
            self._button(row, "检测1", "/segment", {"segment": "inspection_1"})
            self._button(row, "Arm2运动", "/segment", {"segment": "arm2_motion"})
            self._button(row, "检测2", "/segment", {"segment": "inspection_2"})
            self._button(row, "Stop", "/stop", {})
            self._button(row, "Continue", "/continue", {})
            self._button(row, "Reset", "/reset", {})
            root.addWidget(controls)

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

            progress_group = QGroupBox("进度、检测与 KPI")
            progress_layout = QVBoxLayout(progress_group)
            self.progress = QLabel("fins 0/4 | paths 0/8")
            self.arm2_tool = QLabel("Arm2 tool: none")
            self.inspection = QLabel("inspection: -")
            self.kpi = QLabel("KPI: -")
            for label in (self.progress, self.arm2_tool, self.inspection, self.kpi):
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                progress_layout.addWidget(label)
            self.plot = TemperaturePlot()
            progress_layout.addWidget(self.plot)
            root.addWidget(progress_group, 1)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(150)

        def _button(self, layout: Any, text: str, path: str, payload: dict[str, Any]) -> None:
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, p=path, body=payload: self.post(p, body))
            layout.addWidget(button)

        def post(self, path: str, payload: dict[str, Any]) -> None:
            try:
                post_json(base_url + path, payload, timeout=0.5)
            except Exception as exc:
                self.result.setText(f"request failed: {exc}")

        def refresh(self) -> None:
            try:
                state = get_json(base_url + "/state", timeout=0.5)
                self.order.setText(f"order: {state.get('order_id') or '-'}")
                paused = " (PAUSED)" if state.get("paused", False) else ""
                self.stage.setText(f"stage: {state.get('stage', 'IDLE')}{paused}")
                fixture = state.get("fixture", {})
                self.fixture.setText(f"fixture: {fixture.get('status', fixture or '-')}")
                furnace = state.get("furnace", {})
                temperature = float(furnace.get("temperature_c", 25.0))
                self.furnace.setText(f"furnace: {furnace.get('status', '-')}  {temperature:.1f} °C")
                self.plot.add(temperature)
                self.result.setText(
                    f"result: {state.get('disposition') or '-'}  {state.get('last_error', '')}"
                )
                for name, label in self.arms.items():
                    arm = state.get("arms", {}).get(name, {})
                    label.setText(
                        f"{name}: {arm.get('status', 'idle')}  "
                        f"{arm.get('task_type', '')} {arm.get('task_id', '')}"
                    )
                fins = state.get("fins", {})
                paths = state.get("paths", {})
                fin_done = sum(bool(item.get("inserted", False)) for item in fins.values())
                path_done = sum(bool(item.get("applied", False)) for item in paths.values())
                active_fins = sum(bool(item.get("active", False)) for item in fins.values())
                active_paths = sum(bool(item.get("active", False)) for item in paths.values())
                self.progress.setText(
                    f"fins {fin_done}/{active_fins or 4} | paths {path_done}/{active_paths or 8}"
                )
                arm2_tool = state.get("tools", {}).get("arm2", {})
                arm2_process = state.get("arm2_process", {})
                self.arm2_tool.setText(
                    f"Arm2 tool: {arm2_tool.get('current_tool') or 'none'} | "
                    f"path: {arm2_process.get('current_path') or '-'} | "
                    f"applied: {arm2_process.get('completed_paths', 0)}/"
                    f"{arm2_process.get('total_paths', 0)} | "
                    f"tray carrying: {bool(arm2_process.get('tray_carrying', False))}"
                )
                inspections = state.get("inspections", [])
                self.inspection.setText(f"inspection: {inspections[-1] if inspections else '-'}")
                kpi = state.get("kpi", {})
                self.kpi.setText(
                    f"KPI: elapsed={float(kpi.get('order_elapsed', 0.0)):.1f}s  "
                    f"rework={kpi.get('rework_counts', {})}  score={kpi.get('final_quality_score', '-')}"
                )
            except Exception as exc:
                self.result.setText(f"controller unavailable: {exc}")

    app = QApplication.instance() or QApplication(sys.argv[:1])
    panel = ControlPanel()
    panel.resize(1100, 700)
    panel.show()
    camera = CameraWindow()
    camera.resize(760, 620)
    camera.show()
    # Keep Python references alive for the lifetime of the event loop.
    panel._camera_window = camera  # type: ignore[attr-defined]
    return int(app.exec())


__all__ = ["run_ui_client"]
