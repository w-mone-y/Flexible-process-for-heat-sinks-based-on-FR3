"""Project-owned Windows viewer for the V2 dual-install line."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from math import cos, exp, radians, sin
from typing import Any

RENDER_WIDTH = 1200
RENDER_HEIGHT = 720


def _touchpad_pan_delta(dx: float, dy: float) -> tuple[float, float]:
    return -float(dx), float(dy)


def _prepare_renderer_size(model: Any) -> tuple[int, int]:
    return _prepare_renderer_size_for_viewport(model, RENDER_WIDTH, RENDER_HEIGHT)


def _prepare_renderer_size_for_viewport(model: Any, width: int, height: int) -> tuple[int, int]:
    width = max(1, int(width))
    height = max(1, int(height))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    return width, height


@dataclass(slots=True)
class CameraController:
    """Small, input-device-independent free-camera state."""

    lookat: list[float] = field(default_factory=lambda: [1.75, 0.0, 0.30])
    distance: float = 5.6
    azimuth: float = 140.0
    elevation: float = -25.0

    MIN_DISTANCE = 0.65
    MAX_DISTANCE = 40.0
    MIN_ELEVATION = -89.0
    MAX_ELEVATION = 89.0

    def reset(self) -> None:
        self.lookat[:] = [1.75, 0.0, 0.30]
        self.distance = 5.6
        self.azimuth = 140.0
        self.elevation = -25.0

    def zoom(self, amount: float) -> None:
        """Apply a signed zoom amount; positive values zoom in."""

        self.distance = min(
            self.MAX_DISTANCE,
            max(self.MIN_DISTANCE, self.distance * exp(-float(amount))),
        )

    def orbit(self, dx: float, dy: float) -> None:
        self.azimuth += float(dx) * 0.35
        self.elevation = min(
            self.MAX_ELEVATION,
            max(self.MIN_ELEVATION, self.elevation - float(dy) * 0.25),
        )

    def pan(self, dx: float, dy: float) -> None:
        """Move the target on the ground plane in camera-relative axes."""

        scale = max(0.001, self.distance * 0.0025)
        heading = radians(self.azimuth)
        forward = (cos(heading), sin(heading))
        right = (-sin(heading), cos(heading))
        horizontal = float(dx) * scale
        longitudinal = -float(dy) * scale
        self.lookat[0] += right[0] * horizontal + forward[0] * longitudinal
        self.lookat[1] += right[1] * horizontal + forward[1] * longitudinal

    def apply(self, camera: Any) -> None:
        camera.lookat[:] = self.lookat
        camera.distance = self.distance
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation


def run_windows_viewer(application: Any) -> int:
    """Run the V2 Qt viewer and drive the existing application facade."""

    if sys.platform != "win32":
        raise RuntimeError("the project-owned V2 viewer is Windows-only")

    try:
        import mujoco
        from PySide6.QtCore import QEvent, QRectF, Qt, QTimer
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
    except ImportError as exc:
        print(
            f"Windows V2 viewer requires MuJoCo and PySide6 in the configured Conda environment: {exc}",
            file=sys.stderr,
        )
        return 2

    app = QApplication.instance() or QApplication(sys.argv[:1])
    model, data = application.scene.model, application.scene.data
    renderer: Any | None = None
    camera = mujoco.MjvCamera()
    controller = CameraController()
    controller.apply(camera)

    class ViewerSurface(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setMouseTracking(True)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
            self._last_pos: Any | None = None
            self._frame = QImage()
            self._error = ""
            self._renderer_size = (0, 0)

        def _viewport_pixels(self) -> tuple[int, int]:
            scale = max(1.0, float(self.devicePixelRatioF()))
            width = min(4096, max(1, round(self.width() * scale)))
            height = min(4096, max(1, round(self.height() * scale)))
            return width, height

        def _ensure_renderer(self) -> None:
            nonlocal renderer
            size = self._viewport_pixels()
            if renderer is not None and self._renderer_size == size:
                return
            if renderer is not None:
                renderer.close()
            render_width, render_height = _prepare_renderer_size_for_viewport(model, *size)
            renderer = mujoco.Renderer(model, height=render_height, width=render_width)
            self._renderer_size = size

        def _zoom_from_wheel(self, event: Any) -> bool:
            pixel_delta = event.pixelDelta()
            if not pixel_delta.isNull() and not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                controller.pan(*_touchpad_pan_delta(pixel_delta.x(), pixel_delta.y()))
                return True
            angle_delta = event.angleDelta()
            if angle_delta.y() or angle_delta.x():
                controller.zoom(float(angle_delta.y()) / 240.0)
                return True
            return False

        def event(self, event: Any) -> bool:  # type: ignore[override]
            if event.type() == QEvent.Type.NativeGesture:
                gesture = event.gestureType()
                if gesture == Qt.NativeGestureType.ZoomNativeGesture:
                    controller.zoom(float(event.value()))
                    self.update()
                    event.accept()
                    return True
                if gesture == Qt.NativeGestureType.PanNativeGesture:
                    delta = event.delta()
                    controller.pan(*_touchpad_pan_delta(delta.x(), delta.y()))
                    self.update()
                    event.accept()
                    return True
            return super().event(event)

        def mousePressEvent(self, event: Any) -> None:  # type: ignore[override]
            self._last_pos = event.position()
            self.setFocus()
            event.accept()

        def mouseMoveEvent(self, event: Any) -> None:  # type: ignore[override]
            if self._last_pos is None:
                return
            current = event.position()
            delta = current - self._last_pos
            self._last_pos = current
            buttons = event.buttons()
            if buttons & Qt.MouseButton.LeftButton and not buttons & (
                Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton
            ):
                controller.orbit(float(delta.x()), float(delta.y()))
            elif buttons & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
                controller.pan(float(delta.x()), float(delta.y()))
            self.update()
            event.accept()

        def mouseReleaseEvent(self, event: Any) -> None:  # type: ignore[override]
            self._last_pos = None
            event.accept()

        def wheelEvent(self, event: Any) -> None:  # type: ignore[override]
            if self._zoom_from_wheel(event):
                self.update()
                event.accept()
                return
            event.ignore()

        def keyPressEvent(self, event: Any) -> None:  # type: ignore[override]
            key = event.key()
            if key in (Qt.Key.Key_W, Qt.Key.Key_Up):
                controller.pan(0.0, -40.0)
            elif key in (Qt.Key.Key_S, Qt.Key.Key_Down):
                controller.pan(0.0, 40.0)
            elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
                controller.pan(40.0, 0.0)
            elif key in (Qt.Key.Key_D, Qt.Key.Key_Right):
                controller.pan(-40.0, 0.0)
            elif key == Qt.Key.Key_R:
                controller.reset()
            else:
                super().keyPressEvent(event)
                return
            self.update()
            event.accept()

        def render_frame(self) -> None:
            try:
                self._ensure_renderer()
                controller.apply(camera)
                assert renderer is not None
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                self._frame = QImage(
                    frame.data,
                    int(frame.shape[1]),
                    int(frame.shape[0]),
                    int(frame.strides[0]),
                    QImage.Format.Format_RGB888,
                ).copy()
                self._error = ""
            except Exception as exc:  # OpenGL/driver errors are platform-owned.
                self._error = str(exc)
            self.update()

        def paintEvent(self, event: Any) -> None:  # type: ignore[override]
            del event
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            if self._frame.isNull():
                return
            target = QRectF(self.rect()).adjusted(8.0, 8.0, -8.0, -8.0)
            image_size = self._frame.size()
            image_size.scale(target.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio)
            image_rect = QRectF(
                target.center().x() - image_size.width() / 2.0,
                target.center().y() - image_size.height() / 2.0,
                image_size.width(),
                image_size.height(),
            )
            painter.drawImage(image_rect, self._frame)

        def resizeEvent(self, event: Any) -> None:  # type: ignore[override]
            self._renderer_size = (0, 0)
            super().resizeEvent(event)

    class ViewerWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("V2 双安装线 MuJoCo Viewer")
            self.setMinimumSize(640, 420)
            self.surface = ViewerSurface()
            self.setCentralWidget(self.surface)
            self.timer = QTimer(self)
            self.timer.setTimerType(Qt.TimerType.PreciseTimer)
            self.timer.timeout.connect(self.step)

        def start(self) -> None:
            self.show()
            self.surface.setFocus()
            self.surface.render_frame()
            self.timer.start(max(1, round(1000.0 / application.VIEWER_FPS)))

        def step(self) -> None:
            if not application.controls.running:
                app.quit()
                return
            application.drain_commands()
            application.advance_frame()
            application.publish(viewer_running=True)
            application.update_dual_camera()
            self.surface.render_frame()
            if self.surface._error:
                self.setWindowTitle(f"V2 双安装线 MuJoCo Viewer - {self.surface._error}")
            else:
                self.setWindowTitle("V2 双安装线 MuJoCo Viewer")

        def closeEvent(self, event: Any) -> None:  # type: ignore[override]
            self.timer.stop()
            if renderer is not None:
                renderer.close()
            event.accept()

    window = ViewerWindow()
    application.shared.update(viewer_running=True)
    application.publish(viewer_running=True)
    window.resize(1200, 720)
    window.start()
    return int(app.exec())


__all__ = ["CameraController", "run_windows_viewer"]
